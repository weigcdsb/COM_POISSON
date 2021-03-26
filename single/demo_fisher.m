rng(123)
addpath(genpath('D:\GitHub\COM_POISSON'));

%%
T = 100;
dt = 0.02; % bin length (s)
n = 1; % number of independent observations

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = [[repmat(0, 1, round(T/(dt*2)))...
    repmat(2, 1, T/dt - round(T/(dt*2)))]',...
    [repmat(-1, 1, round(T/(dt*4)))...
    repmat(2, 1, T/dt - round(T/(dt*4)))]'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

spk_vec = zeros(n, T/dt);
for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
end


%% fit
% N = spk_vec;
theta0 = [log(mean(spk_vec(1:100))); 0];
% theta0 = theta_true(1, :)';
% W0 = eye(2);
% F = eye(2);
% Q = dt*eye(2);


[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

[theta_fit2,W_fit2] =...
    ppafilt_fisher_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

[theta_fit3,W_fit3] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

[theta_fit4,W_fit4] =...
    ppasmoo_fisher_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));



theta1 = figure;
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(1, :), 'r', 'LineWidth', 1);

l4 = plot(theta_fit3(1, :), 'c', 'LineWidth', 1);
l5 = plot(theta_fit4(1, :),'color', [0.9290, 0.6940, 0.1250], 'LineWidth', 1);

% plot(theta_fit1(1, :) + sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit1(1, :) - sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit2(1, :) + sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
% plot(theta_fit2(1, :) - sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
legend('true', 'filtering', 'filtering-fisher',...
    'smoothing', 'smoothing-fisher', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
hold off

theta2 = figure;
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(2, :), 'r', 'LineWidth', 1);

l4 = plot(theta_fit3(2, :), 'c', 'LineWidth', 1);
l5 = plot(theta_fit4(2, :),'color', [0.9290, 0.6940, 0.1250], 'LineWidth', 1);

% plot(theta_fit1(2, :) + sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit1(2, :) - sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit2(2, :) + sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
% plot(theta_fit2(2, :) - sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
legend('true', 'filtering', 'filtering-fisher',...
    'smoothing', 'smoothing-fisher', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
hold off
