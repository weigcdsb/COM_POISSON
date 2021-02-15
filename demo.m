rng(123)

%% demo 1: step lambda + step nu
T = 10;
dt = 0.01;
n = 50;

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = [[repmat(4, 1, round(T/(dt*2)))...
    repmat(5, 1, T/dt - round(T/(dt*2)))]',...
    [repmat(4, 1, round(T/(dt*2)))...
    repmat(5, 1, T/dt - round(T/(dt*2)))]'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

figure(999)
plot(nu_true*dt)

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);
theo_mlogy = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k)*dt, nu_true(k)*dt, n);
    cum_app = sum_calc(lam_true(k)*dt, nu_true(k)*dt, 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    Ck = cum_app(4);
    
    theo_mean(k) = Ak/Zk;
    theo_var(k) = Bk/Zk - theo_mean(k)^2;
    theo_mlogy(k) = Ck/Zk;
end

figure(1)
hold on
plot(mean(spk_vec, 1))
plot(theo_mean)
hold off

figure(2)
hold on
plot(var(spk_vec, 1))
plot(theo_var)
hold off

figure(3)
hold on
plot(mean(gammaln(spk_vec + 1), 1))
plot(theo_mlogy)
hold off

[theta_fit1,W_fit1] =...
    ppafilt_compoisson(spk_vec,X_lam,G_nu,theta_true(1, :)',...
    eye(2),eye(2),dt*eye(2),dt);

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson(spk_vec,X_lam,G_nu,theta_true(1, :)',...
    eye(2),eye(2),dt*eye(2),dt);

lam_fit1 = exp(X_lam.*theta_fit1(1, :)');
lam_fit2 = exp(X_lam.*theta_fit2(1, :)');
nu_fit1 = exp(G_nu.*theta_fit1(2, :)');
nu_fit2 = exp(G_nu.*theta_fit2(2, :)');

lambda = figure;
hold on
plot(lam_true*dt, 'r', 'LineWidth', 2)
plot(lam_fit1*dt, 'b', 'LineWidth', 2)
plot(lam_fit2*dt, 'g', 'LineWidth', 2)
legend('true', 'filtering', 'smoothing', 'Location','northwest')
title('\lambda\Delta t')
xlabel('step')
hold off
saveas(lambda, 'lambda_1.png')

nu = figure;
hold on
plot(nu_true*dt, 'r', 'LineWidth', 2)
plot(nu_fit1*dt, 'b', 'LineWidth', 2)
plot(nu_fit2*dt, 'g', 'LineWidth', 2)
legend('true', 'filtering', 'smoothing', 'Location','northwest')
title('\nu\Delta t')
xlabel('step')
hold off
saveas(nu, 'nu_1.png')

%% demo 2: step lambda + cos nu
clear all; close all; clc;
T = 10;
dt = 0.01;
n = 50;

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

period = T/(2*dt);
theta_true = [[repmat(3.5, 1, round(T/(dt*2)))...
    repmat(4.5, 1, T/dt - round(T/(dt*2)))]',...
    3+2*cos((10*pi/period)*(1:T/dt))'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

spk_vec = zeros(n, T/dt);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k)*dt, nu_true(k)*dt, n);
end

[theta_fit1,W_fit1] =...
    ppafilt_compoisson(spk_vec,X_lam,G_nu,theta_true(1, :)',...
    eye(2),eye(2),dt*eye(2),dt);

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson(spk_vec,X_lam,G_nu,theta_true(1, :)',...
    eye(2),eye(2),dt*eye(2),dt);

lam_fit1 = exp(X_lam.*theta_fit1(1, :)');
lam_fit2 = exp(X_lam.*theta_fit2(1, :)');
nu_fit1 = exp(G_nu.*theta_fit1(2, :)');
nu_fit2 = exp(G_nu.*theta_fit2(2, :)');

lambda = figure;
hold on
plot(lam_true*dt, 'r', 'LineWidth', 2)
plot(lam_fit1*dt, 'b', 'LineWidth', 2)
plot(lam_fit2*dt, 'g', 'LineWidth', 2)
legend('true', 'filtering', 'smoothing', 'Location','northwest')
title('\lambda\Delta t')
xlabel('step')
hold off
saveas(lambda, 'lambda_2.png')

nu = figure;
hold on
plot(nu_true*dt, 'r', 'LineWidth', 2)
plot(nu_fit1*dt, 'b', 'LineWidth', 2)
plot(nu_fit2*dt, 'g', 'LineWidth', 2)
legend('true', 'filtering', 'smoothing', 'Location','northwest')
title('\nu\Delta t')
xlabel('step')
hold off
saveas(nu, 'nu_2.png')