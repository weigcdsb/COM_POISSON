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

% % Case 3 -- Mean increase + Var decrease
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
% Elapsed time is 0.675340 seconds.

% filtering: window = 20
tic;
theta_filt2 =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 20, windType);
toc;
% Elapsed time is 1.338085 seconds.

% filtering: window = 100
tic;
theta_filt3 =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 100, windType);
toc;
% Elapsed time is 3.790261 seconds.


% Newton-Raphson: Fisher hessian
nStep = size(spk_vec, 2);
tic;
gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-6,1000);
theta_newton = reshape(theta_newton_vec, [], nStep);
toc;
% Elapsed time is 1.994968 seconds.

% Newton-Raphson: exact hessian
tic;
gradHess_tmp = @(vecTheta) gradHessThetaExact(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton2_vec,~,~,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-6,1000);
theta_newton2 = reshape(theta_newton2_vec, [], nStep);
toc;
% Elapsed time is 1.319081 seconds.


subplot(1,2,1)
hold on
plot(theta_true(:,1),'k')
plot(theta_filt(1,:),'r')
plot(theta_newton2(1,:),'b')
plot(theta_filt2(1,:),'g')
plot(theta_filt3(1,:),'c')
title('\beta')
hold off

subplot(1,2,2)
hold on
plot(theta_true(:,2),'k')
plot(theta_filt(2,:),'r')
plot(theta_newton2(2,:),'b')
plot(theta_filt2(2,:),'g')
plot(theta_filt3(2,:),'c')
title('\gamma')
legend('true', 'smoother', 'newton', 'window-20', 'window-100')
hold off

%% use filtering results as warm start
tic;
gradHess_tmp = @(vecTheta) gradHessThetaExact(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton2_vec,~,niSigTheta_newton,~] = newtonGH(gradHess_tmp,theta_filt(:),1e-6,1000);
theta_newton2 = reshape(theta_newton2_vec, [], nStep);
toc;
% Elapsed time is 2.022655 seconds.

