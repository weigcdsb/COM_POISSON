addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
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


% coarse bin
% bin = 2;
% t = zeros(ceil(length(t_raw)/bin), 1);
% pos = zeros(ceil(length(pos_raw)/bin), 1);
% posAlign = zeros(ceil(length(pos_raw)/bin), 1);
% spk = zeros(ceil(length(pos_raw)/bin), 1);
% for k = 1:ceil(length(pos_raw)/bin)
%     raw_idx = (bin*(k-1) + 1):min((bin*k), length(pos_raw));
%     t(k) = mean(t_raw(raw_idx));
%     pos(k) = mean(pos_raw(raw_idx));
%     posAlign(k) = mean(posAlign_raw(raw_idx));
%     spk(k) = sum(spk_raw(raw_idx));
% end

% no coarse bin
bin = 1;
t = t_raw;
pos = pos_raw;
posAlign = posAlign_raw;
spk = spk_raw;

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

subplot(1, 3, 1)
plot(spk)
title('spikes')

subplot(1, 3, 2)
hold on
% plot(spk, 'Color', [1, 0.5, 0, 0.2])
plot(mean_smoo, 'b', 'LineWidth', 1)
title('window-mean')
hold off

subplot(1, 3, 3)
hold on
plot(var_smoo./mean_smoo, 'b', 'LineWidth', 1)
title('window-fano factor')
hold off

%%

nknots=20;
% Gnknots=1;
Gnknots=1;
X = getCubicBSplineBasis((pos + 250)/500,nknots,false);
G = getCubicBSplineBasis((pos + 250)/500,Gnknots,false);

% X = getCubicBSplineBasis((pos + 250)/500*2*pi,nknots,true);
% G = getCubicBSplineBasis((pos + 250)/500*2*pi,Gnknots,true);



%% sanity check the place field to make sure nknots is reasonable...
vispos = linspace(0,1,256);
visX = getCubicBSplineBasis(vispos,nknots,false);
figure(3)

% clf
% pos_cir_tmp = position_circular(idx0:end,1);
% subplot(3,2,1:2)
% plot(t,(pos_cir_tmp-1)*cam2cm,'k')
% hold on
% plot(t(spike_counts(idx0:end,neuron)>0),(pos_cir_tmp(spike_counts(idx0:end,neuron)>0,1)-1)*cam2cm,'r.')
% box off
% hold off
% axis tight


subplot(1,2,1)
plot(pos,spk,'.')
b = glmfit(X,spk,'poisson','constant','off');
hold on
plot(vispos'*500-250,exp(visX*b))
hold off
axis tight

subplot(1,2,2)
plot(vispos'*500-250,visX)
axis tight

%%
% nCMP = round(2000/bin); % 500
nCMP = find(run_number>5,1); % fit with some min # of runs down the track
writematrix(spk(1:nCMP), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\y.csv')
writematrix(X(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\X.csv')
writematrix(G(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\G.csv')

% basedir = 'C:\Users\ian\Documents\GitHub\';
% writematrix(spk(1:nCMP), [basedir 'COM_POISSON\demo\hc\y.csv'])
% writematrix(X(1:nCMP, :), [basedir 'COM_POISSON\demo\hc\X.csv'])
% writematrix(G(1:nCMP, :), [basedir 'COM_POISSON\demo\hc\G.csv'])

%% fit
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmpreg.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmp_t1.csv');

% RunRcode([basedir 'COM_POISSON\demo\hc\cmpreg.r'],...
%     'C:\Program Files\R\R-4.0.2\bin');
% theta0 = readmatrix([basedir 'COM_POISSON\demo\hc\cmp_t1.csv']);

windType = 'forward';

% Q = diag([repmat(1e-3,nknots,1); repmat(1e-3,Gnknots,1)]); % single  X & G
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-3,Gnknots,1)]); % single G
% Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-4,Gnknots + 1,1)]); % multi G

% fit smoother twice
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk', X, G,...
    eye(length(theta0)),eye(length(theta0)),Q, 10, windType); % initial: use window 10/20?

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

f = @(Q) helper_window_v2(Q, theta02, spk(1:nSub1)',X(1:nSub1, :),G(1:nSub1, :),...
    W02,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

% Q_lam = Qopt(1);
Q_lam = [Qopt(1) Qopt(2)*ones(1, size(X, 2)-1)]; % multi

Q_nu = Qopt(min(size(X, 2), 2)+1); % single
% Q_nu = [Qopt(min(size(X, 2), 2)+1) Qopt(min(size(X, 2), 2)+2)*ones(1, size(G, 2) - 1)]; % multi
Qoptmatrix = diag([Q_lam Q_nu]);


% window selection
windSize0 = 5;
searchStep = 5;
windUB = 150;
windSet = [1 windSize0:searchStep:windUB];
nSearchMax = length(windSet);
nSub2 = round(length(spk)/5);

fWind = @(windSize) helper_window_v2_windSize(windSize, Qoptmatrix,...
    theta02, spk(1:nSub2)', X(1:nSub2, :), G(1:nSub2, :), W02,...
    eye(length(theta0)), windType, searchStep);

llhd_filt = [];
nDec = 0;
llhd_pre = -Inf;

for k = 1:nSearchMax
    
    llhd_tmp = -fWind(windSet(k));
    llhd_filt = [llhd_filt llhd_tmp];
    if(llhd_tmp < llhd_pre)
        nDec = nDec + 1;
    else
        nDec = 0;
    end
    llhd_pre = llhd_tmp;    
    
    if nDec > 2
        break
    end  
end

plot(windSet(1:k), llhd_filt)
[~, winIdx] = max(llhd_filt);
optWinSize = windSet(winIdx);

[theta_fit,W_fit] =...
    ppasmoo_compoisson_v2_window_fisher(theta02, spk', X, G,...
    W02,eye(length(theta0)),Qoptmatrix, optWinSize, windType);

save('C:\Users\gaw19004\Desktop\COM_POI_data\hc_12_QTune_noBin_20_1_1.mat')

subplot(1,2,1)
plot(theta_fit(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit((nknots+2):end, :)')
title('gamma')

% % double single
% subplot(1,2,1)
% plot(theta_fit(1:(nknots), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit((nknots+1):end, :)')
% title('gamma')

%%
lam = exp(sum(X .* theta_fit(1:(nknots+1), :)',2));
nu = exp(sum(G .* theta_fit((nknots+2):end, :)',2));

% % double single
% lam = exp(sum(X .* theta_fit(1:(nknots), :)',2));
% nu = exp(sum(G .* theta_fit((nknots+1):end, :)',2));


CMP_mean = zeros(size(lam, 1), 1);
CMP_var = zeros(size(lam, 1), 1);
logZ = zeros(size(lam, 1), 1);
for k = 1:length(lam)
    logcum_app = logsum_calc(lam(k), nu(k), 500);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ(k) = log_Z;
    CMP_mean(k) = exp(log_A - log_Z);
    CMP_var(k) = exp(log_B - log_Z) - CMP_mean(k)^2;
end


subplot(1, 2, 1)
hold on
plot(spk, 'b', 'LineWidth', 1)
% plot(mean_smoo, 'b', 'LineWidth', 1)
plot(CMP_mean, 'r', 'LineWidth', 1)
title('obs-mean')
hold off

subplot(1, 2, 2)
hold on
plot(var_smoo./mean_smoo, 'b', 'LineWidth', 1)
plot(CMP_var./CMP_mean, 'r', 'LineWidth', 1)
title('fano factor')
hold off

llhd = sum(spk.*log((lam+(lam==0))) -...
        nu.*gammaln(spk + 1) - logZ);
llhd/sum(spk)


%% 

[posMin, posMax] = bounds((pos + 250)/500);
heatmap_pos = linspace(posMin,posMax,256);
heatmap_X = getCubicBSplineBasis(heatmap_pos,nknots,false);
heatmap_G = getCubicBSplineBasis(heatmap_pos,Gnknots,false);

heatmap_lam = exp(heatmap_X * theta_fit(1:(nknots+1), :));
heatmap_nu = exp(heatmap_G * theta_fit((nknots+2):end, :));

skip = 50; % resolution
heatmap_lam = heatmap_lam(:,1:skip:end);
heatmap_nu = heatmap_nu(:,1:skip:end);
heatmap_t = t_raw(1:skip:end);

heatmap_mean = zeros(size(heatmap_lam));
heatmap_var = zeros(size(heatmap_lam));

for t = 1:size(heatmap_lam,2)
    for i=1:size(heatmap_lam,1)
        logcum_app = logsum_calc(heatmap_lam(i,t), heatmap_nu(i,t), 500);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
    
        heatmap_mean(i,t) = exp(log_A - log_Z);
        heatmap_var(i,t) = exp(log_B - log_Z) - heatmap_mean(i,t)^2;
    end
end

%%
colIdx = find(max(heatmap_mean, [], 1) < max(spk_raw(:))*2);
heatmap_mean = heatmap_mean(:, colIdx);
heatmap_var = heatmap_var(:, colIdx);
heatmap_t = heatmap_t(colIdx);

figure(20)
subplot(3,1,1)
plot(t_raw,pos)
hold on
scatter(t_raw(find(spk_raw>0)),pos(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
hold off
box off; set(gca,'TickDir','out')
ylim([-250 250])
xlim([0 t_raw(end)])

subplot(3,1,2)
imagesc(heatmap_t,heatmap_pos*500-250,heatmap_mean)
box off; set(gca,'TickDir','out')
set(gca,'YDir','normal')
ylabel('Mean')
% set(gca,'CLim',[0 max(spk_raw(:))*2])
colorbar

subplot(3,1,3)
imagesc(heatmap_t,heatmap_pos*500-250,log10(heatmap_var./heatmap_mean))
box off; set(gca,'TickDir','out')
set(gca,'YDir','normal')
ylabel('log Fano Factor')
xlabel('Time [min]')
% set(gca,'CLim',[0 20])
colorbar

%%
% %% heatmap -- spikes
% t = t_raw;
% [tMin, tMax] = bounds(t);
% [posMin, posMax] = bounds(posAlign);
% 
% subplot(1,2,1)
% % % interpolate directly...
% % [xq,yq] = meshgrid(tMin:0.01:tMax, posMin:.1:posMax);
% % vq = griddata(t,posAlign,spk,xq,yq); %'natural'/ 'linear'
% % % [vq,~,~] = gridfit(t,posAlign,spk,...
% % %     tMin:0.1:tMax,posMin:.1:posMax);
% % imagesc([tMin, tMax], [posMin, posMax], vq)
% % set(gca,'YDir','normal') 
% % colorbar
% 
% % interpolate the line, bin time
% tWind = 0.4;
% nT = floor(range(t)/tWind);
% posAlignq = posMin:.1:posMax;
% SpkMatrix = zeros(length(posAlignq), nT);
% 
% for k = 1:nT
%     
%     idx_tmp = (tWind*(k-1)+ tMin <= t) & (t < min(tWind*k+ tMin, tMax));
%     spk_tmp = spk(idx_tmp);
%     posAlign_tmp = posAlign(idx_tmp);
%     
%     [posAlign_tmp2, ~, n] = unique(posAlign_tmp , 'first');
%     spk_tmp2  = accumarray(n , spk_tmp , size(posAlign_tmp2) , @(x) sum(x));
%     
%     if(isempty(spk_tmp)); continue; end;
%     
%     if(length(spk_tmp2) == 1)
%         [~, minIdx] = min(abs(posAlign_tmp2 - posAlignq));
%         SpkMatrix(minIdx, k) = spk_tmp2;
%     else
%         SpkMatrix(:, k) = interp1(posAlign_tmp2,spk_tmp2,posAlignq, 'linear', 0);
%     end
%     
% end
% 
% imagesc([tMin, tMax], [posMin, posMax], SpkMatrix)
% set(gca,'YDir','normal') 
% colorbar
% caxis([0 20]);
% 
% subplot(1,2,2)
% % interpolate the line, bin time
% tWind = 0.4;
% nT = floor(range(t)/tWind);
% posAlignq = posMin:.1:posMax;
% SpkMatrix = zeros(length(posAlignq), nT);
% 
% for k = 1:nT
%     idx_tmp = (tWind*(k-1)+ tMin <= t) & (t < min(tWind*k+ tMin, tMax));
%     spk_tmp = CMP_mean(idx_tmp);
%     posAlign_tmp = posAlign(idx_tmp);
%     
%     [posAlign_tmp2, ~, n] = unique(posAlign_tmp , 'first');
%     spk_tmp2  = accumarray(n , spk_tmp , size(posAlign_tmp2) , @(x) sum(x));
%     
%     if(isempty(spk_tmp)); continue; end;
%     
%     if(length(spk_tmp2) == 1)
%         [~, minIdx] = min(abs(posAlign_tmp2 - posAlignq));
%         SpkMatrix(minIdx, k) = spk_tmp2;
%     else
%         SpkMatrix(:, k) = interp1(posAlign_tmp2,spk_tmp2,posAlignq, 'linear', 0);
%     end
%     
% end
% 
% imagesc([tMin, tMax], [posMin, posMax], SpkMatrix)
% set(gca,'YDir','normal') 
% colorbar
% caxis([0 20]);
% 
% 
% %% heatmap -- FF
% [tMin_smoo, tMax_smoo] = bounds(t_smoo);
% [posMin_smoo, posMax_smoo] = bounds(posAlign_smoo);
% 
% subplot(1, 2, 1)
% tWind = 0.4;
% nT_smoo = floor(range(t_smoo)/tWind);
% posAlignq_smoo = posMin_smoo:.1:posMax_smoo;
% ffMatrix_smoo = zeros(length(posAlignq_smoo), nT_smoo);
% ff_smoo = var_smoo./mean_smoo;
% 
% for k = 1:nT_smoo
%     idx_tmp = (tWind*(k-1)+ tMin_smoo <= t_smoo) & (t_smoo < min(tWind*k+ tMin_smoo, tMax_smoo));
%      
%     ff_tmp = ff_smoo(idx_tmp);
%     posAlign_tmp = posAlign_smoo(idx_tmp);
%     
%     [posAlign_tmp2, ~, n] = unique(posAlign_tmp , 'first');
%     ff_tmp2  = accumarray(n , ff_tmp , size(posAlign_tmp2) , @(x) sum(x));
%     
%     if(isempty(ff_tmp)); continue; end;
%     
%     if(length(ff_tmp2) == 1)
%         [~, minIdx] = min(abs(posAlign_tmp2 - posAlignq_smoo));
%         ffMatrix_smoo(minIdx, k) = ff_tmp2;
%     else
%         ffMatrix_smoo(:, k) = interp1(posAlign_tmp2,ff_tmp2,posAlignq_smoo, 'linear', 0);
%     end
%     
% end
% 
% imagesc([tMin_smoo, tMax_smoo], [posMin_smoo, posMax_smoo], ffMatrix_smoo)
% set(gca,'YDir','normal') 
% colorbar
% caxis([0 20]);
% 
% 
% subplot(1, 2, 2)
% tWind = 0.4;
% nT = floor(range(t)/tWind);
% posAlignq = posMin:.1:posMax;
% ffMatrix = zeros(length(posAlignq), nT);
% CMP_ff = CMP_var./CMP_mean;
% 
% for k = 1:nT
%     idx_tmp = (tWind*(k-1)+ tMin <= t) & (t < min(tWind*k+ tMin, tMax));
%      
%     
%     ff_tmp = CMP_ff(idx_tmp);
%     posAlign_tmp = posAlign(idx_tmp);
%     
%     [posAlign_tmp2, ~, n] = unique(posAlign_tmp , 'first');
%     ff_tmp2  = accumarray(n , ff_tmp , size(posAlign_tmp2) , @(x) sum(x));
%     
%     if(isempty(ff_tmp)); continue; end;
%     
%     if(length(ff_tmp2) == 1)
%         [~, minIdx] = min(abs(posAlign_tmp2 - posAlignq));
%         ffMatrix(minIdx, k) = ff_tmp2;
%     else
%         ffMatrix(:, k) = interp1(posAlign_tmp2,ff_tmp2,posAlignq, 'linear', 0);
%     end
%     
% end
% 
% imagesc([tMin, tMax], [posMin, posMax], ffMatrix)
% set(gca,'YDir','normal') 
% colorbar
% caxis([0 20]);


