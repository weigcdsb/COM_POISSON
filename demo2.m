rng(12)
%%
clear all; close all; clc;
T = 10;
dt = 0.01;
n = 50;

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

period = T/(2*dt);
theta_true = [[repmat(4, 1, round(T/(dt*2)))...
    repmat(5, 1, T/dt - round(T/(dt*2)))]',...
    5+cos((10*pi/period)*(1:T/dt))'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

spk_vec = zeros(n, T/dt);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k)*dt, nu_true(k)*dt, n);
end

[theta_fit1,W_fit1] =...
    ppafilt_compoisson(spk_vec,X_lam,G_nu,...
    eye(2),eye(2),dt*eye(2),dt);

[theta_fit2,W_fit2] =...
    ppafilt_fisher_compoisson(spk_vec,X_lam,G_nu,...
    eye(2),eye(2),dt*eye(2),dt);

[theta_fit3,W_fit3] =...
    ppasmoo_compoisson(spk_vec,X_lam,G_nu,...
    eye(2),eye(2),dt*eye(2),dt);

[theta_fit4,W_fit4] =...
    ppasmoo_fisher_compoisson(spk_vec,X_lam,G_nu,...
    eye(2),eye(2),dt*eye(2),dt);


lam_fit1 = exp(X_lam.*theta_fit1(1, :)');
lam_fit2 = exp(X_lam.*theta_fit2(1, :)');
lam_fit3 = exp(X_lam.*theta_fit3(1, :)');
lam_fit4 = exp(X_lam.*theta_fit4(1, :)');

nu_fit1 = exp(G_nu.*theta_fit1(2, :)');
nu_fit2 = exp(G_nu.*theta_fit2(2, :)');
nu_fit3 = exp(G_nu.*theta_fit3(2, :)');
nu_fit4 = exp(G_nu.*theta_fit4(2, :)');


theta1 = figure;
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(1, :), 'c', 'LineWidth', 1);

l4 = plot(theta_fit3(1, :), 'r', 'LineWidth', 1);
l5 = plot(theta_fit4(1, :),'color', [0.9290, 0.6940, 0.1250], 'LineWidth', 1);

% plot(theta_fit1(1, :) + sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit1(1, :) - sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit2(1, :) + sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
% plot(theta_fit2(1, :) - sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
legend('true', 'filtering', 'filterin-fisher',...
    'smoothing', 'smoothing-fisher', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
hold off
saveas(theta1, 'theta1_fisher.png')

theta2 = figure;
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(2, :), 'c', 'LineWidth', 1);

l4 = plot(theta_fit3(2, :), 'r', 'LineWidth', 1);
l5 = plot(theta_fit4(2, :),'color', [0.9290, 0.6940, 0.1250], 'LineWidth', 1);

% plot(theta_fit1(2, :) + sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit1(2, :) - sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
% plot(theta_fit2(2, :) + sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
% plot(theta_fit2(2, :) - sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
legend('true', 'filtering', 'filterin-fisher',...
    'smoothing', 'smoothing-fisher', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
hold off
saveas(theta2, 'theta2_fisher.png')
