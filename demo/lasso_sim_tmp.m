addpath(genpath('D:\GitHub\COM_POISSON'));
% addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(1)
T = 100;
dt = 0.02; % bin length (s)
n = 1; % number of independent observations

nx = 10;
ng = 10;

X_lam = normrnd(1,.3,[round(T/dt),nx]);
G_nu = normrnd(1,.3,[round(T/dt),ng]);

% X_lam = ones(T/dt, nx);
% G_nu = ones(T/dt, ng);

theta_true = [[repmat(0, 1, round(T/(dt*2)))...
    repmat(1, 1, T/dt - round(T/(dt*2)))]',...
    zeros(T/dt, nx - 1),...
    [repmat(1, 1, round(T/(dt*4)))...
    repmat(2, 1, T/dt - round(T/(dt*4)))]',...
    zeros(T/dt, ng - 1)];

lam_true = exp(X_lam.*theta_true(:, 1:nx));
nu_true = exp(G_nu.*theta_true(:, (nx+1):end));

spk_vec = zeros(n, T/dt);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
end

%%
% theta0 = [log(mean(spk_vec(1:100))); 0];
theta0 = theta_true(1, :)';
% W0 = eye(2);
% F = eye(2);
% Q = dt*eye(2);
ntheta = length(theta0);

[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(ntheta),eye(ntheta),eye(ntheta)*1e-4);

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(ntheta),eye(ntheta),eye(ntheta)*1e-4);

[theta_fit3,W_fit3] =...
    ppasmoo_compoisson_v2_l1(theta0, spk_vec,X_lam,G_nu,...
    eye(ntheta),eye(ntheta),eye(ntheta)*1e-4, 0.1);

[theta_fit4,W_fit4] =...
    ppafilt_compoisson_v2_l1(theta0, spk_vec,X_lam,G_nu,...
    eye(ntheta),eye(ntheta),eye(ntheta)*1e-4, 0.1);

%% regular filtering/ smoothing is fine enough...
% penalized version is non-stable...

theta_filter = theta_fit1;
W_filter = W_fit1;
theta_smooth = theta_fit2;
W_smooth = W_fit2;

for idx = 1:length(theta0)
    figure(idx)
    hold on
    l1 = plot(theta_true(:, idx), 'k', 'LineWidth', 2);
    l2 = plot(theta_filter(idx, :), 'b', 'LineWidth', 1);
    l3 = plot(theta_smooth(idx, :), 'c', 'LineWidth', 1);
    plot(theta_filter(idx, :) + sqrt(squeeze(W_fit1(idx, idx, :)))', 'b:', 'LineWidth', .5)
    plot(theta_filter(idx, :) - sqrt(squeeze(W_fit1(idx, idx, :)))', 'b:', 'LineWidth', .5)
    plot(theta_smooth(idx, :) + sqrt(squeeze(W_fit2(idx, idx, :)))', 'c:', 'LineWidth', .5)
    plot(theta_smooth(idx, :) - sqrt(squeeze(W_fit2(idx, idx, :)))', 'c:', 'LineWidth', .5)
    legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
    title('\theta for \lambda')
    xlabel('step')
    hold off
end


