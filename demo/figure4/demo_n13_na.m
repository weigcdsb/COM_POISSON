addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

% full
trial_x_full=repmat(theta',size(data.EVENTS,2),1);
trial_y_full=[];
c=1;
stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y_full(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
        end
        t=t+stim_length;
        c=c+1;
    end
end

% delete NA theta
trial_y_full = trial_y_full(~isnan(trial_x_full),:);
trial_x_full = trial_x_full(~isnan(trial_x_full));
theta = theta(~isnan(theta));


% sub
rng(1)
nAll = length(theta);
nSS = round(nAll/2);

ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
for k = 1:size(data.EVENTS, 2)
    ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
        sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
end

trial_y_half = trial_y_full*nan;
trial_y_half(ssIdx,:) = trial_y_full(ssIdx,:);
trial_y_narm = trial_y_full(ssIdx, :);

neuron=13;
spk_vec_full = trial_y_full(:,neuron)';
spk_vec_half = trial_y_half(:,neuron)';
spk_vec_narm = trial_y_narm(:, neuron)';
T = length(trial_x_full);

Xnknots = 5;
Gnknots = 3;

Xb = getCubicBSplineBasis(trial_x_full,Xnknots,true);
Gb = getCubicBSplineBasis(trial_x_full,Gnknots,true);

%% fit1: full data
initIdx = 5*length(theta);
writematrix(spk_vec_full(1:initIdx)', [r_wd '\y.csv'])
writematrix(Xb(1:initIdx, :),[r_wd '\X.csv'])
writematrix(Gb(1:initIdx, :),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = diag([repmat(1e-4,Xnknots+1,1); repmat(1e-4,Gnknots + 1,1)]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec_full, Xb, Gb,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);
theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(Q, theta01, spk_vec_full,Xb,Gb,...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, Xb,Gb, theta01, W01,...
    eye(length(theta01)), Qoptmatrix1, spk_vec_full);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit1 = reshape(theta_newton_vec, [], T);

lam1 = zeros(1,T);
nu1 = zeros(1,T);
CMP_mean1 = zeros(1, T);
CMP_var1 = zeros(1, T);
logZ1 = zeros(1,T);

for m = 1:T
    lam1(m) = exp(Xb(m,:)*theta_fit1(1:(Xnknots+1), m));
    nu1(m) = exp(Gb(m,:)*theta_fit1((Xnknots+2):end, m));
    [CMP_mean1(m), CMP_var1(m), ~, ~, ~, logZ1(m)] = ...
            CMPmoment(lam1(m), nu1(m), 1000);
end

CMP_ff1 = CMP_var1./CMP_mean1;

%% fit2: half data

Xb_narm = Xb(ssIdx,:);
Gb_narm = Gb(ssIdx,:);

initIdx = max(5*nSS, find(cumsum(spk_vec_narm) > 200, 1, 'first'));
writematrix(spk_vec_narm(1:initIdx)', [r_wd '\y.csv'])
writematrix(Xb_narm(1:initIdx, :),[r_wd '\X.csv'])
writematrix(Gb_narm(1:initIdx, :),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = diag([repmat(1e-4,Xnknots+1,1); repmat(1e-4,Gnknots + 1,1)]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec_half, Xb, Gb,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);
theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(Q, theta02, spk_vec_half,Xb,Gb,...
    W02,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb, 2) - 1)];
Qoptmatrix2 = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, Xb,Gb, theta02, W02,...
    eye(length(theta02)), Qoptmatrix2, spk_vec_half);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], T);

lam2 = zeros(1,T);
nu2 = zeros(1,T);
CMP_mean2 = zeros(1, T);
CMP_var2 = zeros(1, T);
logZ2 = zeros(1,T);

for m = 1:T
    lam2(m) = exp(Xb(m,:)*theta_fit2(1:(Xnknots+1), m));
    nu2(m) = exp(Gb(m,:)*theta_fit2((Xnknots+2):end, m));
    [CMP_mean2(m), CMP_var2(m), ~, ~, ~, logZ2(m)] = ...
            CMPmoment(lam2(m), nu2(m), 1000);
end

CMP_ff2 = CMP_var2./CMP_mean2;

%% plotting...
spk_mat = reshape(spk_vec_full,[],size(data.EVENTS,2));
cmp_mean1_mat = reshape(CMP_mean1,[],size(data.EVENTS,2));
cmp_mean2_mat = reshape(CMP_mean2,[],size(data.EVENTS,2));
[theta_sort,sid]=sort(theta);

CMP_ff1_mat = reshape(CMP_ff1,[],size(data.EVENTS,2));
CMP_ff2_mat = reshape(CMP_ff2,[],size(data.EVENTS,2));


x0 = linspace(0,2*pi,256)';
bas = getCubicBSplineBasis(x0,Xnknots,true);

% spk_mean = mean(spk_mat(sid,:), 2);
% po1 = find(spk_mean == max(spk_mean(1:size(spk_mat,1)/2)));
% po2 = find(spk_mean == max(spk_mean((size(spk_mat,1)/2+1):end)));
% plot(spk_mean)

cmp_mean_mean = mean(cmp_mean1_mat(sid,:), 2);
po1 = find(cmp_mean_mean == max(cmp_mean_mean(1:size(cmp_mean_mean,1)/2)));
po2 = find(cmp_mean_mean == max(cmp_mean_mean((size(cmp_mean_mean,1)/2+1):end)));


plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure4';
cd(plotFolder)

spk_plot = figure;
% theta_sort: rad
imagesc(1:size(data.EVENTS, 2),57.2958*theta_sort,spk_mat(sid,:))
colormap(flipud(hot));
colorbar()
hold on
% yline(theta(theta_po1), 'b--', 'LineWidth', 4);
% yline(theta(theta_po2), 'c--', 'LineWidth', 4);
yline(57.2958*theta_sort(po1), 'b--', 'LineWidth', 4);
yline(57.2958*theta_sort(po2), 'c--', 'LineWidth', 4);
hold off
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(spk_plot,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(spk_plot, '1_spk.svg')
saveas(spk_plot, '1_spk.png')


FR_full = figure;
imagesc(1:size(data.EVENTS, 2),57.2958*theta_sort,cmp_mean1_mat(sid,:))
cLim = caxis;
colormap(flipud(hot));
% colormap(hot)
hold on
yline(57.2958*theta_sort(po1), 'b--', 'LineWidth', 4);
yline(57.2958*theta_sort(po2), 'c--', 'LineWidth', 4);
hold off
colorbar()
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_full,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(FR_full, '2_FR_full.svg')
saveas(FR_full, '2_FR_full.png')

FR_half = figure;
imagesc(1:size(data.EVENTS, 2),57.2958*theta_sort,cmp_mean2_mat(sid,:))
set(gca,'CLim',cLim)
colormap(flipud(hot));
hold on
yline(57.2958*theta_sort(po1), 'b--', 'LineWidth', 4);
yline(57.2958*theta_sort(po2), 'c--', 'LineWidth', 4);
hold off
colorbar()
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_half,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(FR_half, '3_FR_half.svg')
saveas(FR_half, '3_FR_half.png')


smoothing=15;
mff=[]; mmm=[]; mffse=[]; mmmse=[];
for i=1:(120-smoothing)
    ff=[];
    c=1;
    for stim=[theta_sort(po1) theta_sort(po2)]
        tv=[];
        tv = find(abs(theta-stim)<20*pi/180);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):nPos:(nPos*smoothing)]+i*nPos];
        end
        ff = getFF(trial_y_full(tid,neuron),'bayes_bootstrap');
        mff(c,i) = mean(ff);
        mffse(c,i) = std(ff);
        mmm(c,i) = mean(trial_y_full(tid,neuron));
        mmmse(c,i) = std(trial_y_full(tid,neuron))/sqrt(length(tid));
        c=c+1;
    end
end


FF = figure;
subplot(2,1,1)
plot([1:(120-smoothing)]+smoothing/2,mff(1,:),'Color','b')
hold on
plot(CMP_ff1_mat(theta_po1,:), 'r--')
plot(CMP_ff2_mat(theta_po1,:), 'g--')
plot([1:(120-smoothing)]+smoothing/2,mff(1,:)'+mffse(1,:)','k:')
plot([1:(120-smoothing)]+smoothing/2,mff(1,:)'-mffse(1,:)','k:')
hold off
ylabel('Fano Factor')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
subplot(2,1,2)
plot([1:(120-smoothing)]+smoothing/2,mff(2,:),'Color','c')
hold on
plot(CMP_ff1_mat(theta_po2,:), 'r--')
plot(CMP_ff2_mat(theta_po2,:), 'g--')
plot([1:(120-smoothing)]+smoothing/2,mff(2,:)'+mffse(2,:)','k:')
plot([1:(120-smoothing)]+smoothing/2,mff(2,:)'-mffse(2,:)','k:')
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
xlabel('Trial')

set(FF,'PaperUnits','inches','PaperPosition',[0 0 4 4])
saveas(FF, '4_FF.svg')
saveas(FF, '4_FF.png')







