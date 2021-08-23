addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%%
rng(123)
T = 100;
dt = 0.1; % bin length (s)
N = 1; % number of independent observations
Q_true = diag([1e-3 1e-5]);


% X_lam = ones(T/dt, 1);
% G_nu = ones(T/dt, 1);
X_lam = normrnd(1,.5,[round(T/dt),1]);
G_nu = normrnd(1,.5,[round(T/dt),1]);

%
beta_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(1, 1))));
gamma_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(2, 2))));

lam_true = exp(X_lam.*beta_true);
nu_true = exp(G_nu.*gamma_true);
spk_vec = com_rnd(lam_true, nu_true);

theta_true = [beta_true gamma_true];

%% fit with different methods

% filtering
max_init = 100;
p = size(theta_true, 2);
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
nStep = size(theta_true, 1);


W0 = eye(p)*1e-1;
F = diag([1 1]);
Q = diag([1e-4 1e-4]);
windSize = 1;
windType = 'forward';

tic;
theta_filt =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize, windType);
toc;
% Elapsed time is 0.425509 seconds.

% Newton-Raphson
tic;
gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0_tmp, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,niSigTheta_newton,~] = newtonGH(gradHess_tmp,repmat(theta0_tmp,nStep,1),1e-6,1000);
theta_newton = reshape(theta_newton_vec, [], nStep);
toc;
% Elapsed time is 0.912050 seconds.

subplot(1,2,1)
hold on
plot(theta_true(:,1),'k')
plot(theta_filt(1,:),'r')
plot(theta_newton(1,:),'b')
title('\beta')
hold off

subplot(1,2,2)
hold on
plot(theta_true(:,2),'k')
plot(theta_filt(2,:),'r')
plot(theta_newton(2,:),'b')
title('\gamma')
legend('true', 'filter')
hold off




