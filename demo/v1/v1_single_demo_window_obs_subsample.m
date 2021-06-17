addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

%% samples

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

% sub
rng(1)

nAll = length(theta);
nSS = 20; % round(nAll/2)
ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
for k = 1:size(data.EVENTS, 2)
    ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
        sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
end

trial_x_ss = trial_x_full(ssIdx);
trial_y_ss = trial_y_full(ssIdx, :);

%% EDA
neuron=13;
nplot = 10;

[~,theta_idx]=sort(theta);

smoothing=10;
mff=[]; mmm=[];
obs_ff = []; obs_mean = [];
for i=1:size(data.EVENTS, 2)-smoothing
    
    c=1;
    for j = theta_idx
        tid = [j:nAll:(nAll*smoothing)]+i*nAll;
        ff = getFF(trial_y_full(tid, neuron));
        obs_ff(c, i) = ff;
        obs_mean(c, i) = mean(trial_y_full(tid, neuron));
        c = c+1;
    end
    
    ff=[];
    c=1;
    for stim=linspace(0,2*pi,nplot)
        tv=[];
        tv = find(abs(theta-stim)<pi/5);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):nAll:(nAll*smoothing)]+i*nAll];
        end
        ff = getFF(trial_y_full(tid,neuron));
        mff(c,i) = mean(ff);
        mmm(c,i) = mean(trial_y_full(tid,neuron));
        c=c+1;
    end
end

subplot(2,2,1)
plot([1:(120-smoothing)]+smoothing/2,mmm')
xlabel('Trial')
ylabel('Mean')
subplot(2,2,2)
plot([1:(120-smoothing)]+smoothing/2,mff')
xlabel('Trial')
ylabel('Fano Factor')
subplot(2,2,3)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
colorbar()
title('Mean-obs')
subplot(2,2,4)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
title('Fano Factor')
colorbar()

% trial_x = trial_x_full;
% trial_y = trial_y_full;
% nObs = nAll;
trial_x = trial_x_ss;
trial_y = trial_y_ss;
nObs = nSS;


%%
ry = reshape(trial_y(:,neuron),nObs,[]);

%
nknots=7;
Gnknots=7;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

trial = 1:10;
writematrix(reshape(ry(:, trial), [], 1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(Gb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\G.csv')

%% fit the data
windType = 'forward';

% cmp initials
% RunRcode('D:\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
%     'E:\software\R\R-4.0.2\bin');
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

% Q-tune
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-4,Gnknots + 1,1)]); % multi G

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, trial_y(:,neuron)', Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q, 5, windType); % initial: use window 5?

theta021 = theta_fit_tmp(:, 1);
W021 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)))];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2(Q, theta021, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb,...
    W021,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

%%
% Qopt = 1e-3*ones(1, 4); % for no Q tune
Q_lam = [Qopt(1) Qopt(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt(3) Qopt(4)*ones(1, size(Gb, 2) - 1)];
Qoptmatrix = diag([Q_lam Q_nu]);
%%

% window selection
winSizeSet = [1 linspace(10, 60, 6)];
np = size(Xb, 2) + size(Gb, 2);
theta0_winSize = zeros(np, length(winSizeSet));
W0_winSize = zeros(np, np, length(winSizeSet));

preLL_winSize_pred = zeros(1, length(winSizeSet));
preLL_winSize_filt = zeros(1, length(winSizeSet));
preLL_winSize_smoo = zeros(1, length(winSizeSet));

idx = 1;
spk_vec = trial_y(:,neuron)';
for k = winSizeSet
%     [theta_fit_tmp,W_fit_tmp] =...
%         ppasmoo_compoisson_v2_window_fisher(theta0, trial_y(:,neuron)', Xb, Gb,...
%         eye(length(theta0)),eye(length(theta0)),Qoptmatrix, k, windType);
%     or just use theta021 & W021
%     
%     theta0_winSize(:, idx) = theta_fit_tmp(:, 1);
%     W0_winSize(:, :, idx) = W_fit_tmp(:, :, 1);
    
    theta0_winSize(:, idx) = theta021;
    W0_winSize(:, :, idx) = W021;
    
    [~,~,lam_pred,nu_pred,log_Zvec_pred,...
        lam_filt,nu_filt,log_Zvec_filt,...
        lam_smoo,nu_smoo,log_Zvec_smoo] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_winSize(:, idx), trial_y(:,neuron)', Xb, Gb,...
        W0_winSize(:, :, idx),eye(length(theta0)),Qoptmatrix, k, windType);
    
    
    if(length(log_Zvec_pred) == size(spk_vec, 2))
        preLL_winSize_pred(idx) = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log_Zvec_pred);
        preLL_winSize_filt(idx) = sum(spk_vec.*log((lam_filt+(lam_filt==0))) -...
            nu_filt.*gammaln(spk_vec + 1) - log_Zvec_filt);
        preLL_winSize_smoo(idx) = sum(spk_vec.*log((lam_smoo+(lam_smoo==0))) -...
            nu_smoo.*gammaln(spk_vec + 1) - log_Zvec_smoo);
    else
        preLL_winSize_pred(idx) = -Inf;
        preLL_winSize_filt(idx) = -Inf;
        preLL_winSize_smoo(idx) = -Inf;
    end
    idx = idx + 1;
end

subplot(1, 3, 1)
plot(winSizeSet, preLL_winSize_pred)
title('prediction')
subplot(1, 3, 2)
plot(winSizeSet, preLL_winSize_filt)
title('filtering')
subplot(1, 3, 3)
plot(winSizeSet, preLL_winSize_smoo)
title('smoothing')

[~, winIdx] = max(preLL_winSize_filt);
optWinSize = winSizeSet(winIdx);
theta0_opt = theta0_winSize(:, winIdx);
W0_opt = W0_winSize(:, :, winIdx);

% formal fit
[theta_fit,W_fit] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_opt, trial_y(:,neuron)', Xb, Gb,...
    W0_opt,eye(length(theta0_opt)),Qoptmatrix, optWinSize, windType);

save('C:\Users\gaw19004\Desktop\COM_POI_data\v1_n13_20.mat')
save('C:\Users\gaw19004\Desktop\COM_POI_data\v1_n13_20_noQTune.mat')
%% parameters track
param = figure;
subplot(1,2,1)
plot(theta_fit(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit((nknots+2):end, :)')
title('gamma')

%% compare to obs.: heatmap (only for full data)
% lam = exp(sum(Xb .* theta_fit(1:(nknots+1), :)',2));
% nu = exp(sum(Gb .* theta_fit((nknots+2):end, :)',2));
% 
% CMP_mean = 0*lam;
% CMP_var = 0*lam;
% 
% for m = 1:length(lam)
%     logcum_app  = logsum_calc(lam(m), nu(m), 1000);
%     log_Z = logcum_app(1);
%     log_A = logcum_app(2);
%     log_B = logcum_app(3);
%     
%     CMP_mean(m) = exp(log_A - log_Z);
%     CMP_var(m) = exp(log_B - log_Z) - CMP_mean(m)^2;
%     
% end
% 
% CMP_ff = CMP_var./CMP_mean;
% ry_hat = reshape(CMP_mean,nObs,[]);
% ry_var = reshape(CMP_var,nObs,[]);
% 
% subplot(2,2,1)
% imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
% title('Mean-obs. smooth')
% caxis([min([obs_mean ry_hat], [], 'all') max([obs_mean ry_hat], [], 'all')]);
% colorbar
% subplot(2,2,2)
% imagesc(ry_hat(theta_idx,:))
% title('Mean-model')
% caxis([min([obs_mean ry_hat], [], 'all') max([obs_mean ry_hat], [], 'all')]);
% colorbar
% subplot(2,2,3)
% imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
% title('Fano Factor-obs. smooth')
% caxis([min([obs_ff ry_var./ry_hat], [], 'all') max([obs_ff ry_var./ry_hat], [], 'all')]);
% colorbar
% subplot(2,2,4)
% imagesc(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% title('Fano Factor-model')
% caxis([min([obs_ff ry_var./ry_hat], [], 'all') max([obs_ff ry_var./ry_hat], [], 'all')]);
% colorbar



%% compare to obs.: line
x0 = linspace(0,2*pi,nplot);
nMin = 3;

bas = getCubicBSplineBasis(x0,nknots,true);
CMP_mean_line = zeros(length(x0), size(data.EVENTS, 2));
CMP_var_line = zeros(length(x0), size(data.EVENTS, 2));

for i = 1:size(data.EVENTS, 2)
    idx_tmp = ((i-1)*nObs+1):(i*nObs);
    theta_ss = trial_x(idx_tmp);
    for j = 1:length(x0)
        
        [~, sortIdx] = sort(abs(x0(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        %         lam_tmp = exp(bas(j,:)*theta_fit(1:(nknots+1), idx_tmp(id)));
        %         nu_tmp = exp(bas(j,:)*theta_fit((nknots+2):end, idx_tmp(id)));
        %         lam = mean(lam_tmp);
        %         nu = mean(nu_tmp);
        
        lam = exp(bas(j,:)*mean(theta_fit(1:(nknots+1), idx_tmp(id)), 2));
        nu = exp(bas(j,:)*mean(theta_fit((nknots+2):end, idx_tmp(id)), 2));
        
        logcum_app  = logsum_calc(lam, nu, 1000);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        
        CMP_mean_line(j, i) = exp(log_A - log_Z);
        CMP_var_line(j, i) = exp(log_B - log_Z) - CMP_mean_line(j, i)^2;
        
    end
end

CMP_ff_line = CMP_var_line./CMP_mean_line;

subplot(2,2,1)
plot(CMP_mean_line')
subplot(2,2,2)
plot(CMP_ff_line')
subplot(2,2,3)
plot([1:(120-smoothing)]+smoothing/2,mmm')
xlabel('Trial')
ylabel('Mean')
subplot(2,2,4)
plot([1:(120-smoothing)]+smoothing/2,mff')
xlabel('Trial')
ylabel('Fano Factor')

