% addpath(genpath('D:\GitHub\COM_POISSON'));
addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(1)
T = 100;
dt = 0.02; % bin length (s)
n = 1; % number of independent observations

nx = 10;
ng = 10;

X_lam = ones(T/dt, nx);
G_nu = ones(T/dt, ng);

theta_true = [[repmat(0, 1, round(T/(dt*2)))...
    repmat(2, 1, T/dt - round(T/(dt*2)))]',...
    zeros(T/dt, nx - 1),...
    [repmat(-1, 1, round(T/(dt*4)))...
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
    ppasmoo_compoisson_v2_l2(theta0, spk_vec,X_lam,G_nu,...
    eye(ntheta),eye(ntheta),eye(ntheta)*1e-4, 0.01);

theta1 = figure;
subplot(1,2,1)
imagesc(theta_fit2(1:nx, :))
title('lam-fit')
colorbar
subplot(1,2,2)
imagesc(theta_true(:, 1:nx)')
title('lam-true')
colorbar

theta2 = figure;
subplot(1,2,1)
imagesc(theta_fit2((nx + 1):end, :))
title('nu-fit')
colorbar
subplot(1,2,2)
imagesc(theta_true(:, (nx + 1):end)')
title('nu-true')
colorbar




