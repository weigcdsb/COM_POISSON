addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%%
rng(1)
T = 10;
dt = 0.005; % bin length (s)
n = 1; % number of independent observations
t = linspace(0,1,T/dt);

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = zeros(T/dt,2);

% % Case 1 -- Mean increase - poisson model (good)
% theta_true(:,1) = (t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1;
% theta_true(:,2) = 0;
% Q=diag([1e-2 1e-6]);

% Case 2 -- Var decrease - constant(ish) mean (not bad)
target_mean = 10;
theta_true(:,2) = 5*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2);
nu_true = exp(G_nu.*theta_true(:, 2));
% theta_true(:,1) = log(10.^nu_true); % better approximation...
theta_true(:,1) = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true));
Q=diag([1e-3 1e-3]);

% Case 3 -- Mean increase + Var decrease
% theta_true(:,2) = 3*(t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% % theta_true(:,1) = log(matchMean(exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1),nu_true));
% % to run fast... use approximation again
% target_mean = exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1);
% theta_true(:,1) = nu_true.*log(target_mean' + (nu_true - 1)./ (2*nu_true));
% Q=diag([1e-3 1e-3]);


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = com_rnd(lam_true, nu_true);

%% fit with different methods
Q = diag([1e-3 1e-3]);
windType = 'forward';
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0_tmp,F,Q, 1, windType);

theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);

% filtering: no window
tic;
theta_filt =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);
toc;
% case 2: Elapsed time is 0.507953 seconds.
% case 3: Elapsed time is 0.552154 seconds.


% filtering: smaller window
windSize1 = 20;
% windSize1 = 10;
tic;
theta_filt2 =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize1, windType);
toc;
% case 2 (wind = 20): Elapsed time is 1.286366 seconds.
% case 3 (wind = 10): Elapsed time is 1.127127 seconds.


% filtering: larger window
windSize2 = 100;
% windSize2 = 50;
tic;
theta_filt3 =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize2, windType);
toc;
% case 2 (wind = 100): Elapsed time is 3.746579 seconds.
% case 3 (wind = 50): Elapsed time is 2.687056 seconds.


% Newton-Raphson: Fisher hessian
% if theta_fit_tmp is from window > 1, NR will get stuck
% Because when window is turned on, although the fitting is much better,
% it's far from MLE.
nStep = size(spk_vec, 2);
tic;
gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-6,1000);
toc;
theta_newton = reshape(theta_newton_vec, [], nStep);
% case 2: Elapsed time is 2.149235 seconds.
% case 3: Elapsed time is 1.203469 seconds.


% Newton-Raphson: exact hessian
% if theta_fit_tmp is from window > 1, will get stuck
tic;
gradHess_tmp = @(vecTheta) gradHessThetaExact(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton2_vec,~,~,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-6,1000);
toc;
theta_newton2 = reshape(theta_newton2_vec, [], nStep);
% case 2: Elapsed time is 1.394820 seconds.
% case 3: Elapsed time is 1.098722 seconds.


subplot(1,2,1)
hold on
plot(theta_true(:,1),'k')
plot(theta_filt(1,:),'r')
plot(theta_filt2(1,:),'m')
plot(theta_filt3(1,:),'g')
plot(theta_newton(1,:),'b')
plot(theta_newton2(1,:),'c')
title('\beta')
hold off

subplot(1,2,2)
hold on
plot(theta_true(:,2),'k')
plot(theta_filt(2,:),'r')
plot(theta_filt2(2,:),'m')
plot(theta_filt3(2,:),'g')
plot(theta_newton(2,:),'b')
plot(theta_newton2(2,:),'c')
title('\gamma')
lgd = legend('true', 'smoother', "window-"+windSize1,...
    "window-"+windSize2,'newton-fisher','newton-exact');
lgd.FontSize = 6;
hold off

%% llhd & MSE

fitted_theta = [theta_filt; theta_filt2; theta_filt3; theta_newton];
mean_Y = zeros(4, nStep);
log_Z = zeros(4, nStep);
lam_all = zeros(4, nStep);
nu_all = zeros(4, nStep);
for m = 1:4
    fit = fitted_theta(((m-1)*2+1):2*m,:);
    lam_all(m,:) = exp(X_lam'.*fit(1,:));
    nu_all(m,:) = exp(G_nu'.*fit(2,:));
    for t = 1:nStep
        [mean_Y(m,t), ~, ~, ~, ~, log_Z(m,t)] = CMPmoment(lam_all(m,t),...
            nu_all(m,t), 1000);
    end
end

% training llhd/spk & mse
llhd_tr = zeros(m,1);
mse_tr = zeros(m,1);
for m = 1:4
    llhd_tr(m) = sum(spk_vec.*log((lam_all(m,:)+(lam_all(m,:)==0))) -...
        nu_all(m,:).*gammaln(spk_vec + 1) - log_Z(m,:))/sum(spk_vec);
    mse_tr(m) = mean((spk_vec - mean_Y(m,:)).^2);
end

llhd_tr
mse_tr

% case 2:
% llhd_tr =
% 
%    -0.2389
%    -0.2440
%    -0.2439
%    -0.2326
% 
% 
% mse_tr =
% 
%     7.3371
%     8.7116
%     8.6456
%     6.8210




% case 3:
% llhd_tr =
% 
%    -0.3015
%    -0.3036
%    -0.3084
%    -0.2937
% 
% 
% mse_tr =
% 
%     3.9766
%     4.5034
%     4.8317
%     3.2648





% single new dataset
llhd_sing = zeros(m,1);
mse_sing = zeros(m,1);
rng(6);
spk_vec_new = com_rnd(lam_true, nu_true);
for m = 1:4
    llhd_sing(m) = sum(spk_vec_new.*log((lam_all(m,:)+(lam_all(m,:)==0))) -...
        nu_all(m,:).*gammaln(spk_vec_new + 1) - log_Z(m,:))/sum(spk_vec_new);
    mse_sing(m) = mean((spk_vec_new - mean_Y(m,:)).^2);
end

llhd_sing
mse_sing

% case 2:
% llhd_sing =
% 
%    -0.2509
%    -0.2493
%    -0.2458
%    -0.2522
% 
% 
% mse_sing =
% 
%     9.5376
%     9.3424
%     8.8332
%     9.4483


% case 3:
% llhd_sing =
% 
%    -0.3106
%    -0.3124
%    -0.3101
%    -0.3067
% 
% 
% mse_sing =
% 
%     4.7866
%     4.9216
%     4.8994
%     4.2639

% multiple new dataset
rng(8)
nNew = 500;
llhd = zeros(4, nNew);
mse = zeros(4, nNew);
for k = 1:nNew
    spk_vec_new = com_rnd(lam_true, nu_true);
    for m = 1:4
        llhd(m,k) = sum(spk_vec_new.*log((lam_all(m,:)+(lam_all(m,:)==0))) -...
            nu_all(m,:).*gammaln(spk_vec_new + 1) - log_Z(m,:))/sum(spk_vec_new);
        mse(m,k) = mean((spk_vec_new - mean_Y(m,:)).^2);
    end
end

mean(llhd,2)
mean(mse,2)

% csae 2:
% ans =
% 
%    -0.2512
%    -0.2503
%    -0.2466
%    -0.2517
% 
% 
% ans =
% 
%     9.2789
%     9.1391
%     8.6788
%     9.1667


% case 3:
% ans =
% 
%    -0.3102
%    -0.3126
%    -0.3110
%    -0.3055
% 
% 
% ans =
% 
%     4.8025
%     4.9387
%     4.9361
%     4.2386






%% use filtering results as warm start
tic;
gradHess_tmp = @(vecTheta) gradHessThetaExact(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton2_vec,~,niSigTheta_newton,~] = newtonGH(gradHess_tmp,theta_filt(:),1e-6,1000);
theta_newton2 = reshape(theta_newton2_vec, [], nStep);
toc;
% case 2: Elapsed time is 1.452179 seconds.
% case 3: Elapsed time is 1.082235 seconds.

