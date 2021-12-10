addpath(genpath('D:\GitHub\COM_POISSON'));
%
%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

%%
trial_x=repmat(theta',size(data.EVENTS,2),1);

trial_y=[];
c=1;

stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
        end
        t=t+stim_length;
        c=c+1;
    end
end

%%
neuron=13;
% 72 46 13
ry = reshape(trial_y(:,neuron),100,[]);
[~,theta_idx]=sort(theta);

%
nknots=7;
Gnknots=7;
X = getCubicBSplineBasis(theta',nknots,true);
G = getCubicBSplineBasis(theta',Gnknots,true);

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

%%
trial = 1:5;
writematrix(reshape(ry(:, trial), [], 1), 'D:\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(repmat(X, length(trial), 1), 'D:\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(repmat(G, length(trial), 1), 'D:\GitHub\COM_POISSON\runRcode\G.csv')

%% fit the data
windType = 'forward';

% cmp initials
RunRcode('D:\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'E:\software\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots + 1,1)]); % multi G

for m = [1 5 10 20 50 100]
    
    [theta_fit_tmp,W_fit_tmp] =...
        ppasmoo_compoisson_v2_window_fisher(theta0, trial_y(:,neuron)', Xb, Gb,...
        eye(length(theta0)),eye(length(theta0)),Q, m, windType);
    
    theta021 = theta_fit_tmp(:, 1);
    W021 = W_fit_tmp(:, :, 1);
    
    [theta_fit1,W_fit1] =...
        ppasmoo_compoisson_v2_window_fisher(theta021, trial_y(:,neuron)', Xb, Gb,...
        W021,eye(length(theta0)),Q, m, windType);
    
    figure(m)
    subplot(1,2,1)
    plot(theta_fit1(1:(nknots+1), :)')
    title('beta')
    subplot(1,2,2)
    plot(theta_fit1((nknots+2):end, :)')
    title('gamma')
    
    
    theta_fit = theta_fit1;
    Xb = getCubicBSplineBasis(trial_x,nknots,true);
    Gb = getCubicBSplineBasis(trial_x,Gnknots,true);
    
    lam = exp(sum(Xb .* theta_fit(1:(nknots+1), :)',2));
    nu = exp(sum(Gb .* theta_fit((nknots+2):end, :)',2));
    
    CMP_mean = zeros(size(lam, 1), size(lam, 2));
    CMP_var = zeros(size(lam, 1), size(lam, 2));
    
    for n = 1:size(lam, 1)
        for l = 1:size(lam, 2)
            cum_app = sum_calc(lam(n, l), nu(n, l), 500);
            Z = cum_app(1);
            A = cum_app(2);
            B = cum_app(3);
            
            CMP_mean(n, l) = A/Z;
            CMP_var(n, l) = B/Z - CMP_mean(n, l)^2;
        end
    end
    
    CMP_ff = CMP_var./CMP_mean;
    
    figure(m+1)
    ry_hat = reshape(CMP_mean,100,[]);
    ry_var = reshape(CMP_var,100,[]);
    subplot(1,2,1)
    imagesc(ry_hat(theta_idx,:))
    title('Mean')
    subplot(1,2,2)
    imagesc(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
    title('Fano Factor')
    colorbar
    
    
end

%% Q-tune
Q = diag([repmat(1e-5,nknots+1,1); repmat(1e-5,Gnknots + 1,1)]); % multi G

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, trial_y(:,neuron)', Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q, 1, windType);

theta021 = theta_fit_tmp(:, 1);
W021 = W_fit_tmp(:, :, 1);

% Q-tune 1: diag, no constraint
QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, length(theta021));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window(Q, theta021, trial_y(1:6000,neuron)',Xb,Gb,...
    W021,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);

Qoptmatrix1 = diag(Qopt1);

% Q-tune 2: diag, same inside
Q0 = [1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)))];
f = @(Q) helper_window_v2(Q, theta021, trial_y(1:6000,neuron)',Xb,Gb,...
    W021,eye(length(theta0)),1, windType);
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt2(3) Qopt2(4)*ones(1, size(Gb, 2) - 1)];

Qoptmatrix2 = diag([Q_lam Q_nu]);

% Q-tune 3: cov, no constraint in diag
Q0 = [1e-4*ones(1, length(theta021)) 0];
f = @(Q) helper_window_v4(Q, theta021, trial_y(1:6000,neuron)',Xb,Gb,...
    W021,eye(length(theta0)),1, windType);
Qopt3 = fmincon(f,Q0,[],[],[],[],[QLB*ones(1, length(Q0)-1), -QUB],...
    [QUB*ones(1, length(Q0)-1), QUB], [], options);

Qoptmatrix3 = diag(Qopt3(1:(end - 1)));
Qoptmatrix3(1, size(Xb, 2)+1) = Qopt3(end);
Qoptmatrix3(size(Xb, 2)+1, 1) = Qopt3(end);


% Q-tune 4: cov, same inside
Q0 = [1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2))) 0];
f = @(Q) helper_window_v3(Q, theta021, trial_y(1:6000,neuron)',Xb,Gb,...
    W021,eye(length(theta0)),1, windType);

Qopt4 = fmincon(f,Q0,[],[],[],[],[QLB*ones(1, length(Q0)-1), -QUB],...
    [QUB*ones(1, length(Q0)-1), QUB], [], options);

Q_lam = [Qopt4(1) Qopt4(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt4(3) Qopt4(4)*ones(1, size(Gb, 2) - 1)];

Qoptmatrix4 = diag([Q_lam Q_nu]);
Qoptmatrix4(1, length(Q_lam)+1) = Qopt4(end);
Qoptmatrix4(length(Q_lam)+1, 1) = Qopt4(end);



%%
% Qoptmatrix = Qoptmatrix4;
Qoptmatrix = Q;

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
    [theta_fit_tmp,W_fit_tmp] =...
        ppasmoo_compoisson_v2_window_fisher(theta0, trial_y(:,neuron)', Xb, Gb,...
        eye(length(theta0)),eye(length(theta0)),Qoptmatrix, k, windType);
    
    
    theta0_winSize(:, idx) = theta_fit_tmp(:, 1);
    W0_winSize(:, :, idx) = W_fit_tmp(:, :, 1);
    
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

save('v1_window_diagQ.mat')

%% track
param = figure;
subplot(1,2,1)
plot(theta_fit(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit((nknots+2):end, :)')
title('gamma')


%
Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

lam = exp(sum(Xb .* theta_fit(1:(nknots+1), :)',2));
nu = exp(sum(Gb .* theta_fit((nknots+2):end, :)',2));

CMP_mean = zeros(size(lam, 1), size(lam, 2));
CMP_var = zeros(size(lam, 1), size(lam, 2));

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        cum_app = sum_calc(lam(m, n), nu(m, n), 500);
        Z = cum_app(1);
        A = cum_app(2);
        B = cum_app(3);
        
        CMP_mean(m, n) = A/Z;
        CMP_var(m, n) = B/Z - CMP_mean(m, n)^2;
    end
end

CMP_ff = CMP_var./CMP_mean;

%%

heat = figure;
ry_hat = reshape(CMP_mean,100,[]);
ry_var = reshape(CMP_var,100,[]);
subplot(1,2,1)
imagesc(ry_hat(theta_idx,:))
% imagesc(ry_hat(theta_idx,10:end))
title('Mean')
subplot(1,2,2)
imagesc(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% imagesc(ry_var(theta_idx,10:end)./ry_hat(theta_idx,10:end))
title('Fano Factor')
colorbar


%%
line = figure;
subplot(1,2,1)
plot(ry_hat(theta_idx,:))
% plot(ry_hat(theta_idx,10:end))
title('Mean')
subplot(1,2,2)
plot(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% plot(ry_var(theta_idx,10:end)./ry_hat(theta_idx,10:end))
title('Fano Factor')



