rng(123)

%% generate data
T = 10;
dt = 0.01;
n = 50;

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

period = T/(2*dt);

theta_true = [[repmat(3, 1, round(T/(dt*2)))...
    repmat(4, 1, T/dt - round(T/(dt*2)))]',...
    4+2*cos((10*pi/period)*(1:T/dt))'];

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
%% fit

[theta_fit1,W_fit1] =...
    ppafilt_compoisson(spk_vec,X_lam,G_nu,ones(2,1),eye(2),eye(2),1e-5*eye(2),dt);

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson(spk_vec,X_lam,G_nu,ones(2,1),eye(2),eye(2),1e-5*eye(2),dt);

figure(4)
hold on
plot(theta_true(:, 1))
plot(theta_fit1(1, :))
plot(theta_fit2(1, :))
hold off

figure(5)
hold on
plot(theta_true(:, 2))
plot(theta_fit1(2, :))
plot(theta_fit2(2, :))
hold off


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
title('\lambda')
xlabel('step')
hold off
saveas(lambda, 'lambda.png')

nu = figure;
hold on
plot(nu_true*dt, 'r', 'LineWidth', 2)
plot(nu_fit1*dt, 'b', 'LineWidth', 2)
plot(nu_fit2*dt, 'g', 'LineWidth', 2)
legend('true', 'filtering', 'smoothing', 'Location','northwest')
title('\nu')
xlabel('step')
hold off
saveas(nu, 'nu.png')

fit1_mean = zeros(T/dt, 1);
fit1_var = zeros(T/dt, 1);
fit1_mlogy = zeros(T/dt, 1);

fit2_mean = zeros(T/dt, 1);
fit2_var = zeros(T/dt, 1);
fit2_mlogy = zeros(T/dt, 1);

for k = 1:(T/dt)
    cum_app = sum_calc(lam_fit1(k)*dt, nu_fit1(k)*dt, 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    Ck = cum_app(4);
    
    fit1_mean(k) = Ak/Zk;
    fit1_var(k) = Bk/Zk - fit1_mean(k)^2;
    fit1_mlogy(k) = Ck/Zk;
    
    cum_app = sum_calc(lam_fit2(k)*dt, nu_fit2(k)*dt, 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    Ck = cum_app(4);
    
    fit2_mean(k) = Ak/Zk;
    fit2_var(k) = Bk/Zk - fit2_mean(k)^2;
    fit2_mlogy(k) = Ck/Zk;
end

meanY = figure;
hold on
plot(mean(spk_vec, 1), 'color', [0 0 0 0.2])
plot(theo_mean, 'r', 'LineWidth', 2)
plot(fit1_mean, 'b', 'LineWidth', 2)
plot(fit2_mean, 'g', 'LineWidth', 2)
legend('obs.','true', 'filtering', 'smoothing', 'Location','northwest')
title('mean of Y_k')
xlabel('step')
hold off
saveas(meanY, 'meanY.png')

varY = figure;
hold on
plot(var(spk_vec, 1), 'color', [0 0 0 0.2])
plot(theo_var, 'r', 'LineWidth', 2)
plot(fit1_var, 'b', 'LineWidth', 2)
plot(fit2_var, 'g', 'LineWidth', 2)
legend('obs.','true', 'filtering', 'smoothing', 'Location','northwest')
title('var of Y_k')
xlabel('step')
hold off
saveas(varY, 'varY.png')

meanLogYfac = figure;
hold on
plot(mean(gammaln(spk_vec + 1), 1), 'color', [0 0 0 0.2])
plot(theo_mlogy, 'r', 'LineWidth', 2)
plot(fit1_mlogy, 'b', 'LineWidth', 2)
plot(fit2_mlogy, 'g', 'LineWidth', 2)
legend('obs.','true', 'filtering', 'smoothing', 'Location','northwest')
title('mean of log(Y_k!)')
xlabel('step')
hold off
saveas(meanLogYfac, 'meanLogYfac.png')










