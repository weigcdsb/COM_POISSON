addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
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
nSS = round(nAll/2);
% nSS = round(2*nAll/3);

ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
for k = 1:size(data.EVENTS, 2)
    ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
        sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
end

trial_x_ss = trial_x_full(ssIdx);
trial_y_ss = trial_y_full(ssIdx, :);

neuron=13;
% neuron = 11;

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

figure(1)
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
T = length(trial_x);

spk_vec = trial_y(:,neuron)';
ry = reshape(trial_y(:,neuron),nObs,[]);

nknots = 5;
% nknots = 10;

%% check if reasonable
X = getCubicBSplineBasis(trial_x,nknots,true);
x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

[b,dev,stats] = glmfit(X,trial_y(:,neuron),'poisson','constant','off');
[yhat,dylo,dyhi]=glmval(b,bas,'log',stats,'constant','off');


figure(2)
plot(trial_x,trial_y(:,neuron),'.')
hold on
plot(x0,yhat)
plot(x0,yhat-dylo)
plot(x0,yhat+dyhi)
hold off

%% fit 1: adaptive CMP, nBasis for lambda = 5, nBasis for nu = 3

Gnknots_full = 3;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb_full = getCubicBSplineBasis(trial_x,Gnknots_full,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb_full((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

windType = 'forward';
RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);


% Q-tune
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-4,Gnknots_full + 1,1)]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb_full,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q, 1, windType);

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb_full, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
% nSub = round(size(trial_y, 1)/2);
nSub = round(size(trial_y, 1));

f = @(Q) helper_window_v2(Q, theta01, trial_y(1:nSub,neuron)',Xb,Gb_full,...
    W01,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb_full, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, Xb,Gb_full, theta01, W01,...
    eye(length(theta01)), Qoptmatrix1, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit1 = reshape(theta_newton_vec, [], T);

% subplot(1,2,1)
% plot(theta_fit1(1:(nknots+1), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit1((nknots+2):end, :)')
% title('gamma')

lam1 = zeros(1,T);
nu1 = zeros(1,T);
CMP_mean1 = zeros(1, T);
CMP_var1 = zeros(1, T);
logZ1 = zeros(1,T);

for m = 1:T
    lam1(m) = exp(Xb(m,:)*theta_fit1(1:(nknots+1), m));
    nu1(m) = exp(Gb_full(m,:)*theta_fit1((nknots+2):end, m));
    [CMP_mean1(m), CMP_var1(m), ~, ~, ~, logZ1(m)] = ...
            CMPmoment(lam1(m), nu1(m), 1000);
end

CMP_ff1 = CMP_var1./CMP_mean1;

%% fit 2: adaptive CMP, nBasis for nu = 1

Gnknots=1;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

windType = 'forward';
RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);


% Q-tune
Q = eye(length(theta0))*1e-4;
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q, 1, windType);

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
% nSub = round(size(trial_y, 1)/3);

f = @(Q) helper_window_v2(Q, theta02, trial_y(1:nSub,neuron)',Xb,Gb,...
    W02,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)];
Q_nu = Qopt2(3);
Qoptmatrix2 = diag([Q_lam Q_nu]);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, Xb,Gb, theta02, W02,...
    eye(length(theta02)), Qoptmatrix2, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], T);

% subplot(1,2,1)
% plot(theta_fit2(1:(nknots+1), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit2((nknots+2):end, :)')
% title('gamma')

lam2 = zeros(1, T);
nu2 = zeros(1, T);
CMP_mean2 = zeros(1, T);
CMP_var2 = zeros(1, T);
logZ2 = zeros(1,T);

for m = 1:T
    lam2(m) = exp(Xb(m,:)*theta_fit2(1:(nknots+1), m));
    nu2(m) = exp(Gb(m,:)*theta_fit2((nknots+2):end, m));
    [CMP_mean2(m), CMP_var2(m), ~, ~, ~, logZ2(m)] = ...
            CMPmoment(lam2(m), nu2(m), 1000);
end

CMP_ff2 = CMP_var2./CMP_mean2;

%% fit 3: adaptive Poisson

trial = 1:15;
b0 = glmfit(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    reshape(ry(:, trial), [], 1),'poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp(spk_vec,Xb, b0,eye(length(b0))*1e-1,eye(length(b0)),1e-4*eye(length(b0)));

theta03 = theta_fit_tmp(:, 1);
W03 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));

f = @(Q) helper_poisson(Q, theta03, trial_y(1:nSub,neuron)',...
    Xb, W03, eye(length(theta03)));
Qopt3 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(Xb, 2))),QUB*ones(1, min(2, size(Xb, 2))), [], options);
Qoptmatrix3 = diag([Qopt3(1) Qopt3(2)*ones(1, size(Xb, 2)-1)]);

gradHess_tmp = @(vecTheta) gradHessTheta_Poisson(vecTheta, Xb, theta03, W03,...
    eye(length(theta03)), Qoptmatrix3, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit3 = reshape(theta_newton_vec, [], T);
lam3 = exp(sum(Xb .* theta_fit3', 2));

plot(theta_fit3')

%% fit 4: static CMP

writematrix(spk_vec', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_fit4 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');

lam4 = exp(Xb*theta_fit4(1:(nknots+1), :));
nu4 = exp(Gb_full*theta_fit4((nknots+2):end, :));
logZ4 = 0*lam4;

CMP_mean4 = 0*lam4;
CMP_var4 = 0*lam4;

for m = 1:length(lam4)
    logcum_app  = logsum_calc(lam4(m), nu4(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ4(m) = log_Z;
    CMP_mean4(m) = exp(log_A - log_Z);
    CMP_var4(m) = exp(log_B - log_Z) - CMP_mean4(m)^2;
    
end

CMP_ff4 = CMP_var4./CMP_mean4;

%% fit 5: static Poisson
theta_fit5 = glmfit(Xb,spk_vec,'poisson','constant','off');
lam5 = exp(Xb*theta_fit5);


%% training set llhd
llhd1 = sum(spk_vec.*log((lam1+(lam1==0))) -...
        nu1.*gammaln(spk_vec + 1) - logZ1);
llhd2 = sum(spk_vec.*log((lam2+(lam2==0))) -...
        nu2.*gammaln(spk_vec + 1) - logZ2);
    
llhd3 = sum(-lam3' + log((lam3'+(lam3'==0))).*spk_vec - gammaln(spk_vec + 1));
llhd4 = sum(spk_vec'.*log((lam4+(lam4==0))) -...
        nu4.*gammaln(spk_vec' + 1) - logZ4);
llhd5 = sum(-lam5' + log((lam5'+(lam5'==0))).*spk_vec - gammaln(spk_vec + 1));

%% held-out set... linear interpolation
hoIdx = setdiff(1:length(trial_x_full), ssIdx);
nHo = nAll - nObs;
trial_x_ho = trial_x_full(hoIdx);
spk_vec_ho = trial_y_full(hoIdx,neuron)';
T_ho = length(hoIdx);

Xb_ho = getCubicBSplineBasis(trial_x_ho,nknots,true);
Gb_ho_full = getCubicBSplineBasis(trial_x_ho,Gnknots_full,true);
Gb_ho = getCubicBSplineBasis(trial_x_ho,Gnknots,true);

theta_ho1 = theta_interp1(theta_fit1, ssIdx, hoIdx);
theta_ho2 = theta_interp1(theta_fit2, ssIdx, hoIdx);
theta_ho3 = theta_interp1(theta_fit3, ssIdx, hoIdx);


[lam1_ho, nu1_ho, logZ1_ho, CMP_mean1_ho, CMP_var1_ho] =...
    CMP_seq_calc(theta_ho1, Xb_ho, Gb_ho_full, nknots, 1000);
[lam2_ho, nu2_ho, logZ2_ho, CMP_mean2_ho, CMP_var2_ho] =...
    CMP_seq_calc(theta_ho2, Xb_ho, Gb_ho, nknots, 1000);
lam3_ho = exp(sum(Xb_ho .* theta_ho3', 2));
[lam4_ho, nu4_ho, logZ4_ho, CMP_mean4_ho, CMP_var4_ho] =...
    CMP_seq_calc(repmat(theta_fit4,1,T_ho), Xb_ho, Gb_ho_full, nknots, 1000);
lam5_ho = exp(Xb_ho*theta_fit5);

llhd1_ho = sum(spk_vec_ho.*log((lam1_ho+(lam1_ho==0))) -...
        nu1_ho.*gammaln(spk_vec_ho + 1) - logZ1_ho);
llhd2_ho = sum(spk_vec_ho.*log((lam2_ho+(lam2_ho==0))) -...
        nu2_ho.*gammaln(spk_vec_ho + 1) - logZ2_ho);  
llhd3_ho = sum(-lam3_ho' + log((lam3_ho'+(lam3_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));
llhd4_ho = sum(spk_vec_ho.*log((lam4_ho+(lam4_ho==0))) -...
        nu4_ho.*gammaln(spk_vec_ho + 1) - logZ4_ho);
llhd5_ho = sum(-lam5_ho' + log((lam5_ho'+(lam5_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));


%% llhd per spike

diag([sum(spk_vec) sum(spk_vec_ho)])\...
    [llhd1 llhd2 llhd3 llhd4 llhd5;...
    llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho]

%% some plots
x0 = linspace(0,2*pi,256);
basX = getCubicBSplineBasis(x0,nknots,true);
basG_full = getCubicBSplineBasis(x0,Gnknots_full,true);

T_all = length(trial_x_full);
theta1_all = theta_interp1(theta_fit1, ssIdx, 1:50:T_all);
theta2_all = theta_interp1(theta_fit2, ssIdx, 1:50:T_all);
theta3_all = theta_interp1(theta_fit3, ssIdx, 1:50:T_all);

%%
[mean1, var1, ff1] = cmp_grid(theta1_all, nknots, basX, basG_full, 1000);
[mean2, var2, ff2] = cmp_grid(theta2_all, nknots, basX, ones(256,1), 1000);
mean3 = exp(basX*theta3_all);
[mean4, var4, ff4] = cmp_grid(theta_fit4, nknots, basX, basG_full, 1000);
mean5 = exp(basX*theta_fit5);

figure(3)
subplot(2,2,1)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
colorbar()
title('Mean-obs')
subplot(2,2,2)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
title('FF-obs')
colorbar()
subplot(2,2,3)
imagesc(mean1)
title('Mean-fit')
colorbar()
subplot(2,2,4)
imagesc(ff1)
title('FF-fit')
colorbar()


figure(4)
subplot(2,2,1)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
colorbar()
title('Mean-obs')
subplot(2,2,2)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
title('FF-obs')
colorbar()
subplot(2,2,3)
imagesc(mean2)
title('Mean-fit')
colorbar()
subplot(2,2,4)
imagesc(ff2)
title('FF-fit')
colorbar()


figure(5)
imagesc(mean3)

figure(6)
subplot(1,2,1)
plot(mean4)
subplot(1,2,2)
plot(ff4)

figure(7)
plot(mean5)

