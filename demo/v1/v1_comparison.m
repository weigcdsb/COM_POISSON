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

%%
neuron=13;
% trial_x = trial_x_full;
% trial_y = trial_y_full;
% nObs = nAll;
trial_x = trial_x_ss;
trial_y = trial_y_ss;
nObs = nSS;

spk_vec = trial_y(:,neuron)';

ry = reshape(trial_y(:,neuron),nObs,[]);

%
nknots=7;
Gnknots=7;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

%% fit 1: adaptive CMP

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
Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots + 1,1)]);

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q, 5, windType); % initial: use window 5?

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)))];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2(Q, theta01, trial_y(1:min(6000, size(trial_y, 1)),neuron)',Xb,Gb,...
    W01,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt(1) Qopt(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt(3) Qopt(4)*ones(1, size(Gb, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);

[theta_fit1,W_fit1,~,~,~,~,~,~,...
    lam1,nu1,logZ1] =...
    ppasmoo_compoisson_v2_window_fisher(theta01, spk_vec, Xb, Gb,...
    W01,eye(length(theta01)),Qoptmatrix1, 10, windType);
    
% subplot(1,2,1)
% plot(theta_fit1(1:(nknots+1), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit1((nknots+2):end, :)')
% title('gamma')

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

%% fit 2: adaptive Poisson

trial = 1:15;
b0 = glmfit(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    reshape(ry(:, trial), [], 1),'poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp(spk_vec,Xb, b0,eye(length(b0)),eye(length(b0)),1e-3*eye(length(b0)));

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));

f = @(Q) helper_poisson(Q, theta02, trial_y(1:min(6000, size(trial_y, 1)),neuron)',...
    Xb, W02, eye(length(b0)));
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(Xb, 2))),QUB*ones(1, min(2, size(Xb, 2))), [], options);
Qoptmatrix2 = diag([Qopt(1) Qopt(2)*ones(1, size(Xb, 2)-1)]);

[theta_fit2,W_fit2] = ppasmoo_poissexp(spk_vec, Xb, theta02, W02,eye(length(b0)),Qoptmatrix2);
lam2 = exp(sum(Xb .* theta_fit2', 2));

%% fit 3: static CMP

writematrix(spk_vec', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_fit3 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');

lam3 = exp(Xb*theta_fit3(1:(nknots+1), :));
nu3 = exp(Gb*theta_fit3((nknots+2):end, :));
logZ3 = 0*lam3;

CMP_mean3 = 0*lam3;
CMP_var3 = 0*lam3;

for m = 1:length(lam3)
    logcum_app  = logsum_calc(lam3(m), nu3(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ3(m) = log_Z;
    CMP_mean3(m) = exp(log_A - log_Z);
    CMP_var3(m) = exp(log_B - log_Z) - CMP_mean3(m)^2;
    
end

CMP_ff3 = CMP_var3./CMP_mean3;

%% fit 4: static Poisson
theta_fit4 = glmfit(Xb,spk_vec,'poisson','constant','off');
lam4 = exp(Xb*theta_fit4);

%% training set llhd
llhd1 = sum(spk_vec.*log((lam1+(lam1==0))) -...
        nu1.*gammaln(spk_vec + 1) - logZ1);
llhd2 = sum(-lam2' + log((lam2'+(lam2'==0))).*spk_vec - gammaln(spk_vec + 1));
llhd3 = sum(spk_vec'.*log((lam3+(lam3==0))) -...
        nu3.*gammaln(spk_vec' + 1) - logZ3);
llhd4 = sum(-lam4' + log((lam4'+(lam4'==0))).*spk_vec - gammaln(spk_vec + 1));

subplot(2, 2, 1)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean1, 'b', 'LineWidth', 1)
title('adaptive CMP')
hold off

subplot(2, 2, 2)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(lam2, 'b', 'LineWidth', 1)
title('adaptive Poisson')
hold off

subplot(2, 2, 3)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean3, 'b', 'LineWidth', 1)
title('static CMP')
hold off

subplot(2, 2, 4)
hold on
plot(spk_vec, 'Color', [1, 0.5, 0, 0.2])
plot(lam4, 'b', 'LineWidth', 1)
title('static Poisson')
hold off

%% held-out set
hoIdx = setdiff(1:length(trial_x_full), ssIdx);
nHo = nAll - nObs;
trial_x_ho = trial_x_full(hoIdx);
spk_vec_ho = trial_y_full(hoIdx,neuron)';

nMin = 3;
Xb_ho = getCubicBSplineBasis(trial_x_ho,nknots,true);
Gb_ho = getCubicBSplineBasis(trial_x_ho,Gnknots,true);

lam1_ho = zeros(length(hoIdx), 1);
nu1_ho = 0*lam1_ho;
logZ1_ho = 0*lam1_ho;
CMP_mean1_ho = 0*lam1_ho;
CMP_var1_ho = 0*lam1_ho;

lam2_ho = 0*lam1_ho;

for i = 1:size(data.EVENTS, 2)
    
    idx_train = ((i-1)*nObs+1):(i*nObs);
    idx_ho = ((i-1)*nHo+1):(i*nHo);
    theta_ss = trial_x(idx_train);
    
    for j = idx_ho
        
        [~, sortIdx] = sort(abs(trial_x_ho(j) - theta_ss));
        id = sortIdx(1:nMin);
        
        theta1_tmp = mean(theta_fit1(:, idx_train(id)), 2);
        theta2_tmp = mean(theta_fit2(:, idx_train(id)), 2);
        
        lam1_ho(j) = exp(Xb_ho(j,:)*theta1_tmp(1:(nknots+1)));
        nu1_ho(j) = exp(Gb_ho(j,:)*theta1_tmp((nknots+2):end));
        lam2_ho(j) = exp(Xb_ho(j,:)*theta2_tmp);
        
        logcum_app1  = logsum_calc(lam1_ho(j), nu1_ho(j), 1000);
        logZ1_ho(j) = logcum_app1(1);
        log_A = logcum_app1(2);
        log_B = logcum_app1(3);
        
        CMP_mean1_ho(j) = exp(log_A - logZ1_ho(j));
        CMP_var1_ho(j) = exp(log_B - logZ1_ho(j)) - CMP_mean1_ho(j)^2;
        
    end
    
end

lam3_ho = exp(Xb_ho*theta_fit3(1:(nknots+1)));
nu3_ho = exp(Gb_ho*theta_fit3((nknots+2):end));
logZ3_ho = 0*lam3_ho;
CMP_mean3_ho = 0*lam3_ho;
CMP_var3_ho = 0*lam3_ho;

for m = 1:length(lam3_ho)
    logcum_app  = logsum_calc(lam3_ho(m), nu3_ho(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ3_ho(m) = log_Z;
    CMP_mean3_ho(m) = exp(log_A - log_Z);
    CMP_var3_ho(m) = exp(log_B - log_Z) - CMP_mean3_ho(m)^2;
    
end

lam4_ho = exp(Xb_ho*theta_fit4);


llhd1_ho = sum(spk_vec_ho.*log((lam1_ho'+(lam1_ho'==0))) -...
        nu1_ho'.*gammaln(spk_vec_ho + 1) - logZ1_ho');
llhd2_ho = sum(-lam2_ho' + log((lam2_ho'+(lam2_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));
llhd3_ho = sum(spk_vec_ho'.*log((lam3_ho+(lam3_ho==0))) -...
        nu3_ho.*gammaln(spk_vec_ho' + 1) - logZ3_ho);
llhd4_ho = sum(-lam4_ho' + log((lam4_ho'+(lam4_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));


[llhd1 llhd2 llhd3 llhd4;...
llhd1_ho llhd2_ho llhd3_ho llhd4_ho]


subplot(2, 2, 1)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean1_ho, 'b', 'LineWidth', 1)
title('adaptive CMP')
hold off

subplot(2, 2, 2)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(lam2_ho, 'b', 'LineWidth', 1)
title('adaptive Poisson')
hold off

subplot(2, 2, 3)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(CMP_mean3_ho, 'b', 'LineWidth', 1)
title('static CMP')
hold off

subplot(2, 2, 4)
hold on
plot(spk_vec_ho, 'Color', [1, 0.5, 0, 0.2])
plot(lam4_ho, 'b', 'LineWidth', 1)
title('static Poisson')
hold off

