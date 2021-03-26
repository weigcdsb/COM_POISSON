rng(123)
addpath(genpath('D:\GitHub\COM_POISSON'));

%%
T = 100;
dt = 0.01; % bin length (s)
n = 1; % number of independent observations

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

beta_true = [repmat(2, 1, round(T/(dt*3)))...
    repmat(10, 1, round(T/(dt*3)))...
    repmat(2, 1, T/dt - 2*round(T/(dt*3)))]';

lam_true = exp(X_lam.*beta_true);

% set the mean to be approximate 5
nu_true = log(lam_true)/ log(5);
gamma_true = log(nu_true)./G_nu;

figure(1)
subplot(2,2,1);
plot(lam_true)
subplot(2,2,2);
plot(nu_true)
subplot(2,2,3);
plot(lam_true'.^(1/nu_true))

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
    cum_app = sum_calc(lam_true(k), nu_true(k), 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    Ck = cum_app(4);
    
    theo_mean(k) = Ak/Zk;
    theo_var(k) = Bk/Zk - theo_mean(k)^2;
end

figure(2)
subplot(2,1,1)
hold on
plot(mean(spk_vec, 1))
plot(theo_mean)
hold off

subplot(2,1,2)
plot(theo_var)

theta_true = [beta_true gamma_true];
%%
theta0 = [log(mean(spk_vec(1:100))); 0];
[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([1e-4 1e-3]));

theta = figure;
subplot(2, 1, 1)
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

subplot(2, 1, 2)
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

saveas(theta, 'theta_mean.png')
%%
lam_fit1 = exp(X_lam.*theta_fit1(1, :)');
lam_fit2 = exp(X_lam.*theta_fit2(1, :)');
nu_fit1 = exp(G_nu.*theta_fit1(2, :)');
nu_fit2 = exp(G_nu.*theta_fit2(2, :)');

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);
fit1_mean = zeros(T/dt, 1);
fit1_var = zeros(T/dt, 1);
fit2_mean = zeros(T/dt, 1);
fit2_var = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
    [theo_mean(k), theo_var(k)] = meanVar_cmp(lam_true(k), nu_true(k), 1000);
    [fit1_mean(k), fit1_var(k)] = meanVar_cmp(lam_fit1(k), nu_fit1(k), 1000);
    [fit2_mean(k), fit2_var(k)] = meanVar_cmp(lam_fit2(k), nu_fit2(k), 1000);
    
end

mean_var = figure;
subplot(2,1,1)
hold on
s = plot(spk_vec, 'Color', [1, 0.5, 0, 0.2]);
l1 = plot(theo_mean, 'k', 'LineWidth', 2);
l2 = plot(fit1_mean, 'b', 'LineWidth', 1);
l3 = plot(fit2_mean, 'c', 'LineWidth', 1);
hold off
legend([s l1 l2 l3], 'true-spk', 'true', 'filtering', 'smoothing', 'Location','northeast');
title('mean')
xlabel('step')

subplot(2,1,2)
hold on
l1 = plot(theo_var, 'k', 'LineWidth', 2);
l2 = plot(fit1_var, 'b', 'LineWidth', 1);
l3 = plot(fit2_var, 'c', 'LineWidth', 1);
hold off
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northeast');
title('var')
xlabel('step')

saveas(mean_var, 'mean_var_mean.png')


