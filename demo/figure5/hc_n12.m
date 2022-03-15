addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%
%%
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
neuron = 12;

% 10 12


%% EDA
idx0 = 1;

pos_raw = (position_circular(idx0:end)-1)*cam2cm;
posAlign_raw = position_realigned(idx0:end,1)*cam2cm;
spk_raw = spike_counts(idx0:end,neuron);
t_raw = linspace(0,size(pos_raw,1),size(pos_raw,1))/5/60;
t_raw = t_raw+mean(diff(t_raw))/2;

% no coarse bin
bin = 1;
t = t_raw;
pos = pos_raw;
posAlign = posAlign_raw;
spk = spk_raw;
T = length(spk);

smoothing = 3;
mean_smoo = zeros(length(spk) - smoothing + 1, 1);
var_smoo = zeros(length(spk) - smoothing + 1, 1);
t_smoo = zeros(length(spk) - smoothing + 1, 1);
posAlign_smoo = zeros(length(spk) - smoothing + 1, 1);

for k = 1:(length(spk) - smoothing + 1)
    idx = k:(k + smoothing - 1);
    mean_smoo(k) = mean(spk(idx));
    var_smoo(k) = var(spk(idx));
    t_smoo(k) = mean(t(idx));
    posAlign_smoo(k) = mean(posAlign(idx));
end




%%

nknots=12;
Gnknots=1;
X = getCubicBSplineBasis((pos + 250)/500,nknots,false);
G = getCubicBSplineBasis((pos + 250)/500,Gnknots,false);

% circular: no intercept
% X = getCubicBSplineBasis((pos + 250)/500*2*pi,nknots,true);
% G = getCubicBSplineBasis((pos + 250)/500*2*pi,Gnknots,true);
% X = X(:,2:end);


%% sanity check the place field to make sure nknots is reasonable...
vispos = linspace(0,1,256);
visX = getCubicBSplineBasis(vispos,nknots,false);

% vispos = linspace(0,2*pi,256);
% visX = getCubicBSplineBasis(vispos,nknots,true);
% visX = visX(:,2:end);

plot(pos,spk,'.')
b = glmfit(X,spk,'poisson','constant','off');
hold on
plot(vispos'*500-250,exp(visX*b))
% plot(vispos'/2/pi*500-250,exp(visX*b))
hold off
axis tight

%%
nCMP = find(run_number>2,1); % fit with some min # of runs down the track
writematrix(spk(1:nCMP), [r_wd '\y.csv'])
writematrix(X(1:nCMP, :),[r_wd '\X.csv'])
writematrix(G(1:nCMP, :),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);


Q = eye(length(theta0))*1e-4;
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk', X, G,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);


theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-4*ones(1, min(2, size(X, 2))+ min(2, size(G, 2)))];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
% nSub = round(length(spk)/5);
nSub1 = round(length(spk));

f = @(Q) helper_na(Q, theta02, spk(1:nSub1)',X(1:nSub1, :),G(1:nSub1, :),...
    W02,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

% Q_lam = Qopt(1);
Q_lam = [Qopt(1) Qopt(2)*ones(1, size(X, 2)-1)]; % multi

Q_nu = Qopt(min(size(X, 2), 2)+1); % single
% Q_nu = [Qopt(min(size(X, 2), 2)+1) Qopt(min(size(X, 2), 2)+2)*ones(1, size(G, 2) - 1)]; % multi
Qoptmatrix = diag([Q_lam Q_nu]);


gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, X,G, theta02, W02,...
    eye(length(theta02)), Qoptmatrix, spk');

[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit = reshape(theta_newton_vec, [], T);

%% Let's plot
[posMin, posMax] = bounds((pos + 250)/500);
heatmap_pos = linspace(posMin,posMax,256);
heatmap_X = getCubicBSplineBasis(heatmap_pos,nknots,false);
heatmap_G = getCubicBSplineBasis(heatmap_pos,Gnknots,false);

heatmap_lam = exp(heatmap_X * theta_fit(1:(nknots+1), :));
heatmap_nu = exp(heatmap_G * theta_fit((nknots+2):end, :));

skip = 100; % resolution
heatmap_lam = heatmap_lam(:,1:skip:end);
heatmap_nu = heatmap_nu(:,1:skip:end);
heatmap_t = t_raw(1:skip:end);

heatmap_mean = zeros(size(heatmap_lam));
heatmap_var = zeros(size(heatmap_lam));

for t = 1:size(heatmap_lam,2)
    for i=1:size(heatmap_lam,1)
        [heatmap_mean(i,t), heatmap_var(i,t), ~, ~, ~, ~] = ...
            CMPmoment(heatmap_lam(i,t), heatmap_nu(i,t), 1000);
    end
end

colIdx = find(max(heatmap_mean, [], 1) < max(spk_raw(:))*2);
heatmap_mean = heatmap_mean(:, colIdx);
heatmap_var = heatmap_var(:, colIdx);
heatmap_t = heatmap_t(colIdx);

heatmap_mean_align = reshape(heatmap_mean, size(heatmap_mean,1)/2, []);
for kk = 1:size(heatmap_mean_align,2)
    if mod(kk,2) == 0
        heatmap_mean_align(:,kk) = flip(heatmap_mean_align(:,kk));
    end
end

mean_pos = mean(heatmap_mean_align,2);
m1 = max(mean_pos(1:size(heatmap_mean_align,1)/2));
m2 = max(mean_pos((size(heatmap_mean_align,1)/2+1):end));

p1 = find(mean_pos == m1);
p2 = find(mean_pos == m2);

cd('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure5')

spk_obs = figure;
plot(t_raw,posAlign_raw)
hold on
scatter(t_raw(find(spk_raw>0)),posAlign_raw(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
hold off
box off; set(gca,'TickDir','out')
ylim([0 max(posAlign_raw)])
xlim([0 t_raw(end)])
xlabel('Time [min]')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(spk_obs,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(spk_obs, '1_spk_obs.svg')
saveas(spk_obs, '1_spk_obs.png')


FR = figure;
imagesc(heatmap_t,heatmap_pos*250, heatmap_mean_align)
set(gca,'YDir','normal')
colormap(hot)
% colormap(turbo)
yline(heatmap_pos(2*p1)*250, 'r', 'LineWidth', 2)
yline(heatmap_pos(2*p2)*250, 'w', 'LineWidth', 2)
colorbar
% ylim([0 max(posAlign_raw)])
xlim([0 t_raw(end)])
xlabel('Time [min]')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR, '2_FR_colBar.svg')
saveas(FR, '2_FR_colBar.png')


FR2 = figure;
imagesc(heatmap_t,heatmap_pos*250, heatmap_mean_align)
set(gca,'YDir','normal')
colormap(hot)
% colormap(turbo)
yline(heatmap_pos(2*p1)*250, 'r', 'LineWidth', 2)
yline(heatmap_pos(2*p2)*250, 'w', 'LineWidth', 2)
% ylim([0 max(posAlign_raw)])
xlim([0 t_raw(end)])
xlabel('Time [min]')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR2,'PaperUnits','inches','PaperPosition',[0 0 6 3])
saveas(FR2, '3_FR_noCB.svg')
saveas(FR2, '3_FR_noCB.png')

heatmap_ff_align = reshape(heatmap_var./heatmap_mean, size(heatmap_mean,1)/2, []);
for kk = 1:size(heatmap_ff_align,2)
    if mod(kk,2) == 0
        heatmap_ff_align(:,kk) = flip(heatmap_ff_align(:,kk));
    end
end
heatmap_ff_align_log10 = log10(heatmap_ff_align);

FF = figure;
hold on
plot(heatmap_t,heatmap_ff_align_log10(p1,1:2:end), 'r', 'LineWidth', 2)
plot(heatmap_t,heatmap_ff_align_log10(p1,2:2:end), 'r--', 'LineWidth', 2)
plot(heatmap_t,heatmap_ff_align_log10(p2,1:2:end), 'k', 'LineWidth', 2)
plot(heatmap_t,heatmap_ff_align_log10(p2,2:2:end), 'k--', 'LineWidth', 2)
hold off
ylim([-0.1 max(heatmap_ff_align_log10(:))])
ylabel('log_{10}(Fano Factor)')
xlabel('Time [min]')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FF,'PaperUnits','inches','PaperPosition',[0 0 4 5])
saveas(FF, '4_FF.svg')
saveas(FF, '4_FF.png')


% imagesc(heatmap_t,heatmap_pos*500-250,log10(heatmap_var./heatmap_mean))
% box off; set(gca,'TickDir','out')
% set(gca,'YDir','normal')
% ylabel('log Fano Factor')
% xlabel('Time [min]')
% set(gca,'CLim',[0 20])
% colorbar


%%
% subplot(1,2,1)
% plot(theta_fit(1:(nknots+1), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit((nknots+2):end, :)')
% title('gamma')
% 
% 
% lam = exp(sum(X .* theta_fit(1:(nknots+1), :)',2));
% nu = exp(sum(G .* theta_fit((nknots+2):end, :)',2));
% 
% 
% CMP_mean = zeros(size(lam, 1), 1);
% CMP_var = zeros(size(lam, 1), 1);
% logZ = zeros(size(lam, 1), 1);
% for k = 1:length(lam)
%     [CMP_mean(k), CMP_var(k), ~, ~, ~, logZ(k)] = ...
%             CMPmoment(lam(k), nu(k), 1000);
% end
% 
% 
% subplot(1, 2, 1)
% hold on
% plot(spk, 'b', 'LineWidth', 1)
% plot(CMP_mean, 'r', 'LineWidth', 1)
% title('obs-mean')
% hold off
% 
% subplot(1, 2, 2)
% hold on
% plot(var_smoo./mean_smoo, 'b', 'LineWidth', 1)
% plot(CMP_var./CMP_mean, 'r', 'LineWidth', 1)
% title('fano factor')
% hold off
% 
% llhd = sum(spk.*log((lam+(lam==0))) -...
%         nu.*gammaln(spk + 1) - logZ);
% llhd/sum(spk)


%%
% colIdx = find(max(heatmap_mean, [], 1) < max(spk_raw(:))*2);
% heatmap_mean = heatmap_mean(:, colIdx);
% heatmap_var = heatmap_var(:, colIdx);
% heatmap_t = heatmap_t(colIdx);
% 
% figure(20)
% subplot(3,1,1)
% plot(t_raw,pos)
% hold on
% scatter(t_raw(find(spk_raw>0)),pos(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
% hold off
% box off; set(gca,'TickDir','out')
% ylim([-250 250])
% xlim([0 t_raw(end)])
% 
% subplot(3,1,2)
% imagesc(heatmap_t,heatmap_pos*500-250,heatmap_mean)
% box off; set(gca,'TickDir','out')
% set(gca,'YDir','normal')
% ylabel('Mean')
% % set(gca,'CLim',[0 max(spk_raw(:))*2])
% colorbar
% 
% subplot(3,1,3)
% imagesc(heatmap_t,heatmap_pos*500-250,log10(heatmap_var./heatmap_mean))
% box off; set(gca,'TickDir','out')
% set(gca,'YDir','normal')
% ylabel('log Fano Factor')
% xlabel('Time [min]')
% set(gca,'CLim',[0 20])
% colorbar
% 
% 
% figure(22)
% subplot(3,1,1)
% plot(t_raw,posAlign_raw)
% hold on
% scatter(t_raw(find(spk_raw>0)),posAlign_raw(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
% hold off
% box off; set(gca,'TickDir','out')
% ylim([0 max(posAlign_raw)])
% xlim([0 t_raw(end)])
% 
% 
% 
