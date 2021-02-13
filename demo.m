rng(123)

%% generate data
T = 10;
dt = 0.02;
n = 100;

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

period = T/(2*dt);

theta_true = [[repmat(4.5, 1, round(T/(dt*2)))...
    repmat(5, 1, T/dt - round(T/(dt*2)))]',...
    4+cos((10*pi/period)*(1:T/dt))'];

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

plot(lam_true*dt)
plot(nu_true*dt)

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k)*dt, nu_true(k)*dt, n);
    cum_app = sum_calc(lam_true(k)*dt, nu_true(k)*dt, 1000);
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    
    theo_mean(k) = Ak/Zk;
    theo_var(k) = Bk/Zk - theo_mean(k)^2;
    
end

figure(1)
hold on
plot(sum(spk_vec, 1))
plot(n*theo_mean)
hold off

figure(2)
hold on
plot(var(spk_vec, 1))
plot(theo_var)
hold off

%% fit

[theta_fit,W_fit,lam_fit, nu_fit, iter] =...
    ppafilt_compoisson(spk_vec,X_lam,G_nu,theta_true(1, :)',eye(2),eye(2),1e-6*eye(2),dt);




hold on
plot(theta_true(:, 1))
plot(theta_fit(1, :))
hold off

hold on
plot(theta_true(:, 2))
plot(theta_fit(2, :))
hold off


