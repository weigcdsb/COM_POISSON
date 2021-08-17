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

%% MCMC setting
ng = 1000;
windType = 'forward';
F = diag([1 1]);
windSize = 1;

p = size(theta_true, 2);
nStep = size(theta_true, 1);

% priors
W0 = eye(p)*1e-1;

mu00 = zeros(p,1);
Sig00 = eye(p)*1e2;

nu0 = 4;
sig20 = 1e-4;

% pre-allocation
theta_fit = zeros(p, nStep, ng);
theta0_fit = zeros(p, ng);
Q_fit = zeros(p, p, ng);

% initials
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];

Q_fit(:,:,1) = diag([1e-4 1e-4]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0,F,Q_fit(:,:,1), windSize, windType);
theta_fit(:,:,1) = theta_fit_tmp;
theta0_fit(:,1) = theta_fit_tmp(:, 1);

%% Let's do Gibbs Sampling
for g = 2:10
    disp(g)
    % (1) update state vectors
    % Adaptive smoothing
    theta0_tmp = theta0_fit(:,g-1);
    Q_tmp = Q_fit(:,:,g-1);
    
    [theta_tmp,W_tmp] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
        W0,F,Q_tmp, windSize, windType);
    
    % vecTheta = theta_tmp(:);
    hess_tmp = hessTheta(theta_tmp(:), X_lam,G_nu, W0,...
        F, Q_tmp, spk_vec);
    % use Cholesky decomposition to sample efficiently
    R = chol(-hess_tmp,'lower'); % sparse
    z = randn(length(theta_tmp(:)), 1) + R'*theta_tmp(:);
    thetaSamp = R'\z;
    theta_fit(:,:,g) = reshape(thetaSamp,[], nStep);
    
    % (2) update theta0_fit
    Sig0 = inv(inv(Sig00) + inv(W0));
    mu0 = Sig0*(Sig00\mu00 + W0\theta_fit(:,1,g));
    theta0_fit(:,g) = mvnrnd(mu0, Sig0)';
    
    % (3) update Q: diagonal version
    for k = 1:size(theta_fit, 1)
        
        muTheta = F(k,:)*theta_fit(:,1:(nStep-1),g);
        
        alphq = (nu0 + nStep-1)/2;
        betaq = (nu0*sig20 + sum((theta_fit(k,2:nStep,g) - muTheta).^2))/2;
        Q_fit(k,k,g) = 1/gamrnd(alphq, 1/betaq);
    end
    
    figure(1)
    subplot(1,2,1)
    plot(reshape(Q_fit(1,1,1:g), 1, []))
    subplot(1,2,2)
    plot(reshape(Q_fit(2,2,1:g), 1, []))
    
end

%% diagnose
idx = 200:ng;
mean(Q_fit(:,:,idx), 3)

figure(1)
subplot(1,2,1)
hold on
plot(reshape(Q_fit(1,1,1:g), 1, []))
yline(Q_true(1,1), 'r--', 'LineWidth', 2);
hold off
subplot(1,2,2)
hold on
plot(reshape(Q_fit(2,2,1:g), 1, []))
yline(Q_true(2,2), 'r--', 'LineWidth', 2);
hold off