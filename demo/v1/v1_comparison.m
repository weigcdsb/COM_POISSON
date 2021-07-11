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
nSS = 50; % round(nAll/2)
ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
for k = 1:size(data.EVENTS, 2)
    ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
        sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
end

trial_x_ss = trial_x_full(ssIdx);
trial_y_ss = trial_y_full(ssIdx, :);

% neuron=13;
neuron = 11;
nplot = 10;

[~,theta_idx]=sort(theta);

smoothing=5;
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

spk_vec = trial_y(:,neuron)';
ry = reshape(trial_y(:,neuron),nObs,[]);

nknots = 5;
% nknots = 10;

% check if reasonable
X = getCubicBSplineBasis(trial_x,nknots,true);
x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

[b,dev,stats] = glmfit(X,trial_y(:,neuron),'poisson','constant','off');
[yhat,dylo,dyhi]=glmval(b,bas,'log',stats,'constant','off');


figure(12)
plot(trial_x,trial_y(:,neuron),'.')
hold on
plot(x0,yhat)
plot(x0,yhat-dylo)
plot(x0,yhat+dyhi)
hold off

%% fit 1: adaptive CMP, nBasis for nu = 3

%
% nknots = 20;
Gnknots_full = 5;
% windSize = 30;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb_full = getCubicBSplineBasis(trial_x,Gnknots_full,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(Gb_full((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\G.csv')

windType = 'forward';
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

% Q-tune
Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots_full + 1,1)]);

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb_full,...
    eye(length(theta0)),eye(length(theta0)),Q, 10, windType); % initial: use window 10?

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb_full, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
% nSub = round(size(trial_y, 1)/3);

f = @(Q) helper_window_v2(Q, theta01, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb_full,...
    W01,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb_full, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);

% window selection
windSize0 = 5;
searchStep = 5;
windUB = 150;
windSet = [1 windSize0:searchStep:windUB];
nSearchMax = length(windSet);

fWind = @(windSize) helper_window_v2_windSize(windSize, Qoptmatrix1,...
    theta01, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb_full, W01,...
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
optWinSize1 = windSet(winIdx);


[theta_fit1,W_fit1,~,~,~,~,~,~,...
    lam1,nu1,logZ1] =...
    ppasmoo_compoisson_v2_window_fisher(theta01, spk_vec, Xb, Gb_full,...
    W01,eye(length(theta01)),Qoptmatrix1, optWinSize1, windType);
    
subplot(1,2,1)
plot(theta_fit1(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit1((nknots+2):end, :)')
title('gamma')

CMP_mean1 = 0*lam1;
CMP_var1 = 0*lam1;

for m = 1:length(lam1)
    logcum_app  = logsum_calc(lam1(m), nu1(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    CMP_mean1(m) = exp(log_A - log_Z);
    CMP_var1(m) = exp(log_B - log_Z) - CMP_mean1(m)^2;
    
end

CMP_ff1 = CMP_var1./CMP_mean1;

%% fit 2: adaptive CMP, nBasis for nu = 1

Gnknots=1;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(Gb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\G.csv')

windType = 'forward';
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

% Q-tune
Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots,1)]); % single G

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q, 10, windType); % initial: use window 10?

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2(Q, theta02, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb,...
    W02,eye(length(theta02)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)];
Q_nu = Qopt2(3);
Qoptmatrix2 = diag([Q_lam Q_nu]);


% window selection
windSize0 = 5;
searchStep = 5;
windUB = 150;
windSet = [1 windSize0:searchStep:windUB];
nSearchMax = length(windSet);

fWind = @(windSize) helper_window_v2_windSize(windSize, Qoptmatrix2,...
    theta02, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb, W02,...
    eye(length(theta02)), windType, searchStep);

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
optWinSize2 = windSet(winIdx);


[theta_fit2,W_fit2,~,~,~,~,~,~,...
    lam2,nu2,logZ2] =...
    ppasmoo_compoisson_v2_window_fisher(theta02, spk_vec, Xb, Gb,...
    W02,eye(length(theta02)),Qoptmatrix2, optWinSize2, windType);
    
subplot(1,2,1)
plot(theta_fit2(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit2((nknots+2):end, :)')
title('gamma')

CMP_mean2 = 0*lam2;
CMP_var2 = 0*lam2;

for m = 1:length(lam2)
    logcum_app  = logsum_calc(lam2(m), nu2(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    CMP_mean2(m) = exp(log_A - log_Z);
    CMP_var2(m) = exp(log_B - log_Z) - CMP_mean2(m)^2;
    
end

CMP_ff2 = CMP_var2./CMP_mean2;

%% fit 3: adaptive CMP: constant nu

Gnknots=1;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(Gb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\G.csv')

windType = 'forward';
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

% Q-tune
Q = diag([repmat(1e-3,nknots+1,1); repmat(0,Gnknots,1)]); % single G

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q, 5, windType); % initial: use window 5?

theta03 = theta_fit_tmp(:, 1);
W03 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2_noNu(Q, theta03, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb,...
    W03,eye(length(theta03)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt3 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt3(1) Qopt3(2)*ones(1, size(Xb, 2)-1)];
Qoptmatrix3 = diag([Q_lam 0]);


% window selection
windSize0 = 5;
searchStep = 5;
windUB = 150;
windSet = [1 windSize0:searchStep:windUB];
nSearchMax = length(windSet);

fWind = @(windSize) helper_window_v2_windSize(windSize, Qoptmatrix3,...
    theta03, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb, W03,...
    eye(length(theta03)), windType, searchStep);

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
optWinSize3 = windSet(winIdx);

[theta_fit3,W_fit3,~,~,~,~,~,~,...
    lam3,nu3,logZ3] =...
    ppasmoo_compoisson_v2_window_fisher(theta03, spk_vec, Xb, Gb,...
    W03,eye(length(theta03)),Qoptmatrix3, optWinSize3, windType);
    
subplot(1,2,1)
plot(theta_fit3(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit3((nknots+2):end, :)')
title('gamma')

CMP_mean3 = 0*lam3;
CMP_var3 = 0*lam3;

for m = 1:length(lam3)
    logcum_app  = logsum_calc(lam3(m), nu3(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    CMP_mean3(m) = exp(log_A - log_Z);
    CMP_var3(m) = exp(log_B - log_Z) - CMP_mean3(m)^2;
    
end

CMP_ff3 = CMP_var3./CMP_mean3;

%% fit 4: adaptive Poisson

trial = 1:15;
b0 = glmfit(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    reshape(ry(:, trial), [], 1),'poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp(spk_vec,Xb, b0,eye(length(b0)),eye(length(b0)),1e-3*eye(length(b0)));

theta04 = theta_fit_tmp(:, 1);
W04 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));

f = @(Q) helper_poisson(Q, theta04, trial_y(1:min(6000, size(trial_y, 1)),neuron)',...
    Xb, W04, eye(length(theta04)));
Qopt4 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(Xb, 2))),QUB*ones(1, min(2, size(Xb, 2))), [], options);
Qoptmatrix4 = diag([Qopt4(1) Qopt4(2)*ones(1, size(Xb, 2)-1)]);

[theta_fit4,W_fit4] = ppasmoo_poissexp(spk_vec, Xb, theta04, W04,eye(length(theta04)),Qoptmatrix4);
lam4 = exp(sum(Xb .* theta_fit4', 2));

plot(theta_fit4')

%% fit 5: static CMP

writematrix(spk_vec', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_fit5 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');

lam5 = exp(Xb*theta_fit5(1:(nknots+1), :));
nu5 = exp(Gb_full*theta_fit5((nknots+2):end, :));
logZ5 = 0*lam5;

CMP_mean5 = 0*lam5;
CMP_var5 = 0*lam5;

for m = 1:length(lam5)
    logcum_app  = logsum_calc(lam5(m), nu5(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ5(m) = log_Z;
    CMP_mean5(m) = exp(log_A - log_Z);
    CMP_var5(m) = exp(log_B - log_Z) - CMP_mean5(m)^2;
    
end

CMP_ff5 = CMP_var5./CMP_mean5;

%% fit 6: static Poisson
theta_fit6 = glmfit(Xb,spk_vec,'poisson','constant','off');
lam6 = exp(Xb*theta_fit6);

%% training set llhd
llhd1 = sum(spk_vec.*log((lam1+(lam1==0))) -...
        nu1.*gammaln(spk_vec + 1) - logZ1);
llhd2 = sum(spk_vec.*log((lam2+(lam2==0))) -...
        nu2.*gammaln(spk_vec + 1) - logZ2);
llhd3 = sum(spk_vec.*log((lam3+(lam3==0))) -...
        nu3.*gammaln(spk_vec + 1) - logZ3);
    
llhd4 = sum(-lam4' + log((lam4'+(lam4'==0))).*spk_vec - gammaln(spk_vec + 1));
llhd5 = sum(spk_vec'.*log((lam5+(lam5==0))) -...
        nu5.*gammaln(spk_vec' + 1) - logZ5);
llhd6 = sum(-lam6' + log((lam6'+(lam6'==0))).*spk_vec - gammaln(spk_vec + 1));

% mse1 = mean((spk_vec - CMP_mean1).^2);
% mse2 = mean((spk_vec - CMP_mean2).^2);
% mse3 = mean((spk_vec - CMP_mean3).^2);
% mse4 = mean((spk_vec - lam4').^2);
% mse5 = mean((spk_vec - CMP_mean5').^2);
% mse6 = mean((spk_vec - lam6').^2);

subplot(2, 3, 1)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean1, 'b', 'LineWidth', 1)
title("adaptive CMP: nBasis-G = " + Gnknots_full)
hold off


subplot(2, 3, 2)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean2, 'b', 'LineWidth', 1)
title('adaptive CMP: nBasis-G = 1')
hold off

subplot(2, 3, 3)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean3, 'b', 'LineWidth', 1)
title('adaptive CMP: constant nu')
hold off

subplot(2, 3, 4)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(lam4, 'b', 'LineWidth', 1)
title('adaptive Poisson')
hold off

subplot(2, 3, 5)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean5, 'b', 'LineWidth', 1)
title('static CMP')
hold off

subplot(2, 3, 6)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(lam6, 'b', 'LineWidth', 1)
title('static Poisson')
hold off

%% held-out set
hoIdx = setdiff(1:length(trial_x_full), ssIdx);
nHo = nAll - nObs;
trial_x_ho = trial_x_full(hoIdx);
spk_vec_ho = trial_y_full(hoIdx,neuron)';

nMin =3;
Xb_ho = getCubicBSplineBasis(trial_x_ho,nknots,true);
Gb_ho_full = getCubicBSplineBasis(trial_x_ho,Gnknots_full,true);
Gb_ho = getCubicBSplineBasis(trial_x_ho,Gnknots,true);

lam1_ho = zeros(length(hoIdx), 1);
nu1_ho = 0*lam1_ho;
logZ1_ho = 0*lam1_ho;
CMP_mean1_ho = 0*lam1_ho;
CMP_var1_ho = 0*lam1_ho;

lam2_ho = zeros(length(hoIdx), 1);
nu2_ho = 0*lam2_ho;
logZ2_ho = 0*lam2_ho;
CMP_mean2_ho = 0*lam2_ho;
CMP_var2_ho = 0*lam2_ho;

lam3_ho = zeros(length(hoIdx), 1);
nu3_ho = 0*lam3_ho;
logZ3_ho = 0*lam3_ho;
CMP_mean3_ho = 0*lam3_ho;
CMP_var3_ho = 0*lam3_ho;

lam4_ho = 0*lam1_ho;

for i = 1:size(data.EVENTS, 2)
    
    idx_train = ((i-1)*nObs+1):(i*nObs);
    idx_ho = ((i-1)*nHo+1):(i*nHo);
    theta_ss = trial_x(idx_train);
    
    for j = idx_ho
        
        [~, sortIdx] = sort(abs(trial_x_ho(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        theta1_tmp = mean(theta_fit1(:, idx_train(id)), 2);
        theta2_tmp = mean(theta_fit2(:, idx_train(id)), 2);
        theta3_tmp = mean(theta_fit3(:, idx_train(id)), 2);
        theta4_tmp = mean(theta_fit4(:, idx_train(id)), 2);
        
        lam1_ho(j) = exp(Xb_ho(j,:)*theta1_tmp(1:(nknots+1)));
        nu1_ho(j) = exp(Gb_ho_full(j,:)*theta1_tmp((nknots+2):end));
        lam2_ho(j) = exp(Xb_ho(j,:)*theta2_tmp(1:(nknots+1)));
        nu2_ho(j) = exp(Gb_ho(j,:)*theta2_tmp((nknots+2):end));
        lam3_ho(j) = exp(Xb_ho(j,:)*theta3_tmp(1:(nknots+1)));
        nu3_ho(j) = exp(Gb_ho(j,:)*theta3_tmp((nknots+2):end));
        
        lam4_ho(j) = exp(Xb_ho(j,:)*theta4_tmp);
        
        logcum_app1  = logsum_calc(lam1_ho(j), nu1_ho(j), 1000);
        logZ1_ho(j) = logcum_app1(1);
        log_A = logcum_app1(2);
        log_B = logcum_app1(3);
        CMP_mean1_ho(j) = exp(log_A - logZ1_ho(j));
        CMP_var1_ho(j) = exp(log_B - logZ1_ho(j)) - CMP_mean1_ho(j)^2;
        
        logcum_app2  = logsum_calc(lam2_ho(j), nu2_ho(j), 1000);
        logZ2_ho(j) = logcum_app2(1);
        log_A = logcum_app2(2);
        log_B = logcum_app2(3);
        CMP_mean2_ho(j) = exp(log_A - logZ2_ho(j));
        CMP_var2_ho(j) = exp(log_B - logZ2_ho(j)) - CMP_mean2_ho(j)^2;
        
        logcum_app3  = logsum_calc(lam3_ho(j), nu3_ho(j), 1000);
        logZ3_ho(j) = logcum_app3(1);
        log_A = logcum_app3(2);
        log_B = logcum_app3(3);
        CMP_mean3_ho(j) = exp(log_A - logZ3_ho(j));
        CMP_var3_ho(j) = exp(log_B - logZ3_ho(j)) - CMP_mean3_ho(j)^2;
    end
    
end

lam5_ho = exp(Xb_ho*theta_fit5(1:(nknots+1)));
nu5_ho = exp(Gb_ho_full*theta_fit5((nknots+2):end));
logZ5_ho = 0*lam5_ho;
CMP_mean5_ho = 0*lam5_ho;
CMP_var5_ho = 0*lam5_ho;

for m = 1:length(lam5_ho)
    logcum_app  = logsum_calc(lam5_ho(m), nu5_ho(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ5_ho(m) = log_Z;
    CMP_mean5_ho(m) = exp(log_A - log_Z);
    CMP_var5_ho(m) = exp(log_B - log_Z) - CMP_mean5_ho(m)^2;
    
end

lam6_ho = exp(Xb_ho*theta_fit6);


llhd1_ho = sum(spk_vec_ho.*log((lam1_ho'+(lam1_ho'==0))) -...
        nu1_ho'.*gammaln(spk_vec_ho + 1) - logZ1_ho');
llhd2_ho = sum(spk_vec_ho.*log((lam2_ho'+(lam2_ho'==0))) -...
        nu2_ho'.*gammaln(spk_vec_ho + 1) - logZ2_ho');    
llhd3_ho = sum(spk_vec_ho.*log((lam3_ho'+(lam3_ho'==0))) -...
        nu3_ho'.*gammaln(spk_vec_ho + 1) - logZ3_ho');     
llhd4_ho = sum(-lam4_ho' + log((lam4_ho'+(lam4_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));
llhd5_ho = sum(spk_vec_ho'.*log((lam5_ho+(lam5_ho==0))) -...
        nu5_ho.*gammaln(spk_vec_ho' + 1) - logZ5_ho);
llhd6_ho = sum(-lam6_ho' + log((lam6_ho'+(lam6_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));


% mse1_ho = mean((spk_vec_ho - CMP_mean1_ho').^2);
% mse2_ho = mean((spk_vec_ho - CMP_mean2_ho').^2);
% mse3_ho = mean((spk_vec_ho - CMP_mean3_ho').^2);
% mse4_ho = mean((spk_vec_ho - lam4_ho').^2);
% mse5_ho = mean((spk_vec_ho - CMP_mean5_ho').^2);
% mse6_ho = mean((spk_vec_ho - lam6_ho').^2);


[llhd1 llhd2 llhd3 llhd4 llhd5 llhd6;...
llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho]/120

% [mse1 mse2 mse3 mse4 mse5 mse6;...
%     mse1_ho mse2_ho mse3_ho mse4_ho mse5_ho mse6_ho]


subplot(2, 3, 1)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean1_ho, 'b', 'LineWidth', 1)
title("adaptive CMP: nBasis-G = " + Gnknots_full)
hold off

subplot(2, 3, 2)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean2_ho, 'b', 'LineWidth', 1)
title('adaptive CMP: nBasis-G = 1')
hold off

subplot(2, 3, 3)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean3_ho, 'b', 'LineWidth', 1)
title('adaptive CMP: constant nu')
hold off


subplot(2, 3, 4)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(lam4_ho, 'b', 'LineWidth', 1)
title('adaptive Poisson')
hold off

subplot(2, 3, 5)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean5_ho, 'b', 'LineWidth', 1)
title('static CMP')
hold off

subplot(2, 3, 6)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(lam6_ho, 'b', 'LineWidth', 1)
title('static Poisson')
hold off

save('C:\Users\gaw19004\Desktop\COM_POI_data\v1_n11_5_5_bin.mat')

%%
param = figure;
subplot(1,2,1)
plot(theta_fit1(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit1((nknots+2):end, :)')
title('gamma')


x0 = linspace(0,2*pi,nplot);
nMin = 1;

basX = getCubicBSplineBasis(x0,nknots,true);
basG = getCubicBSplineBasis(x0,Gnknots_full,true);

CMP_mean_line = zeros(length(x0), size(data.EVENTS, 2));
CMP_var_line = zeros(length(x0), size(data.EVENTS, 2));

for i = 1:size(data.EVENTS, 2)
    idx_tmp = ((i-1)*nObs+1):(i*nObs);
    theta_ss = trial_x(idx_tmp);
    for j = 1:length(x0)
        
        [~, sortIdx] = sort(abs(x0(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        lam = exp(basX(j,:)*mean(theta_fit1(1:(nknots+1), idx_tmp(id)), 2));
        nu = exp(basG(j,:)*mean(theta_fit1((nknots+2):end, idx_tmp(id)), 2));
        
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


%%

param = figure;
subplot(1,2,1)
plot(theta_fit2(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit2((nknots+2):end, :)')
title('gamma')


x0 = linspace(0,2*pi,nplot);
nMin = 1;

basX = getCubicBSplineBasis(x0,nknots,true);
basG = getCubicBSplineBasis(x0,Gnknots,true);

CMP_mean_line = zeros(length(x0), size(data.EVENTS, 2));
CMP_var_line = zeros(length(x0), size(data.EVENTS, 2));

for i = 1:size(data.EVENTS, 2)
    idx_tmp = ((i-1)*nObs+1):(i*nObs);
    theta_ss = trial_x(idx_tmp);
    for j = 1:length(x0)
        
        [~, sortIdx] = sort(abs(x0(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        lam = exp(basX(j,:)*mean(theta_fit2(1:(nknots+1), idx_tmp(id)), 2));
        nu = exp(basG(j,:)*mean(theta_fit2((nknots+2):end, idx_tmp(id)), 2));
        
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


%%

param = figure;
subplot(1,2,1)
plot(theta_fit3(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit3((nknots+2):end, :)')
title('gamma')


x0 = linspace(0,2*pi,nplot);
nMin = 1;

basX = getCubicBSplineBasis(x0,nknots,true);
basG = getCubicBSplineBasis(x0,Gnknots,true);

CMP_mean_line = zeros(length(x0), size(data.EVENTS, 2));
CMP_var_line = zeros(length(x0), size(data.EVENTS, 2));

for i = 1:size(data.EVENTS, 2)
    idx_tmp = ((i-1)*nObs+1):(i*nObs);
    theta_ss = trial_x(idx_tmp);
    for j = 1:length(x0)
        
        [~, sortIdx] = sort(abs(x0(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        lam = exp(basX(j,:)*mean(theta_fit3(1:(nknots+1), idx_tmp(id)), 2));
        nu = exp(basG(j,:)*mean(theta_fit3((nknots+2):end, idx_tmp(id)), 2));
        
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



