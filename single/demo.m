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

figure(1)
plot(lam_true)
figure(2)
plot(nu_true)

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);
theo_mlogy = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
    cum_app = sum_calc(lam_true(k), nu_true(k), 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    Ck = cum_app(4);
    
    theo_mean(k) = Ak/Zk;
    theo_var(k) = Bk/Zk - theo_mean(k)^2;
    theo_mlogy(k) = Ck/Zk;
end

figure(3)
hold on
plot(mean(spk_vec, 1))
plot(theo_mean)
hold off

figure(4)
hold on
plot(mean(gammaln(spk_vec + 1), 1))
plot(theo_mlogy)
hold off

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
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

theta1 = figure;
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(1, :), 'c', 'LineWidth', 1);
plot(theta_fit1(1, :) + sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(1, :) - sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(1, :) + sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(1, :) - sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
hold off
saveas(theta1, 'theta1_1.png')


theta2 = figure;
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(2, :), 'c', 'LineWidth', 1);
plot(theta_fit1(2, :) + sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(2, :) - sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(2, :) + sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(2, :) - sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
hold off
saveas(theta2, 'theta2_1.png')

%%
clear all; close all; clc;
T = 100;
dt = 0.02; % bin length (s)
n = 1; % number of independent observations

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = [[repmat(0, 1, round(T/(dt*2)))...
    repmat(2, 1, T/dt - round(T/(dt*2)))]',...
    linspace(-1, 2, T/dt)'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = zeros(n, T/dt);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
end

theta0 = [log(mean(spk_vec(1:100))); 0];

[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

theta1 = figure;
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(1, :), 'c', 'LineWidth', 1);
plot(theta_fit1(1, :) + sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(1, :) - sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(1, :) + sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(1, :) - sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
hold off
saveas(theta1, 'theta1_2.png')

theta2 = figure;
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(2, :), 'c', 'LineWidth', 1);
plot(theta_fit1(2, :) + sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(2, :) - sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(2, :) + sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(2, :) - sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
hold off
saveas(theta2, 'theta2_2.png')

%%
clear all; close all; clc;
T = 100;
dt = 0.02; % bin length (s)
n = 1; % number of independent observations

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);


period = T/(2*dt);
theta_true = [[repmat(0, 1, round(T/(dt*2)))...
    repmat(2, 1, T/dt - round(T/(dt*2)))]',...
    0.5+1.5*cos((10*pi/period)*(1:T/dt))'];


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = zeros(n, T/dt);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
end

theta0 = [log(mean(spk_vec(1:100))); 0];

[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-3 1e-3]));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-3 1e-3]));

theta1 = figure;
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(1, :), 'c', 'LineWidth', 1);
plot(theta_fit1(1, :) + sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(1, :) - sqrt(squeeze(W_fit1(1, 1, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(1, :) + sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(1, :) - sqrt(squeeze(W_fit2(1, 1, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
hold off
saveas(theta1, 'theta1_3.png')

theta2 = figure;
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_fit1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_fit2(2, :), 'c', 'LineWidth', 1);
plot(theta_fit1(2, :) + sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit1(2, :) - sqrt(squeeze(W_fit1(2, 2, :)))', 'b:', 'LineWidth', .5)
plot(theta_fit2(2, :) + sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
plot(theta_fit2(2, :) - sqrt(squeeze(W_fit2(2, 2, :)))', 'c:', 'LineWidth', .5)
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
hold off
saveas(theta2, 'theta2_3.png')


