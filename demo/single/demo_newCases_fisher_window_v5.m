addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
addpath(genpath('D:\github\COM_POISSON'));

%%

rng(1)
T = 10;
dt = 0.005; % bin length (s)
n = 1; % number of independent observations
t = linspace(0,1,T/dt);

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = zeros(T/dt,2);

% % Case 1 -- Mean increase - poisson model (good)
% theta_true(:,1) = (t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1;
% theta_true(:,2) = 0;
% Q=diag([1e-2 1e-6]);

% Case 2 -- Var decrease - constant(ish) mean (not bad)
target_mean = 10;
theta_true(:,2) = 5*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2);
nu_true = exp(G_nu.*theta_true(:, 2));
% theta_true(:,1) = log(10.^nu_true); % better approximation...
theta_true(:,1) = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true));
Q=diag([1e-3 1e-3]);

% % Case 3 -- Mean increase + Var decrease
% theta_true(:,2) = 3*(t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% % theta_true(:,1) = log(matchMean(exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1),nu_true));
% % to run fast... use approximation again
% target_mean = exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1);
% theta_true(:,1) = nu_true.*log(target_mean' + (nu_true - 1)./ (2*nu_true));
% Q=diag([1e-3 1e-3]);


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = com_rnd(lam_true, nu_true);
[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);

windType = 'forward';

%%
Qoptmatrix = diag([1e-3 1e-3]);
optWinSize = 100;
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0_tmp,F,Qoptmatrix, optWinSize, windType);

theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);


[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);
[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);
[est_mean2,est_var2]=getMeanVar(exp(theta_fit2(1,:)),exp(theta_fit2(2,:)));

save('C:\Users\gaw19004\Desktop\COM_POI_data\case2.mat')

figure(2)
plotAll_filtSmoo(spk_vec, X_lam, G_nu, theta_true, theta_fit1, theta_fit2)

%% calculate the varCE
[var_rate_exact1, var_rate_app1] = varCE(X_lam, G_nu, theta_fit1, W_fit1);
[var_rate_exact2, var_rate_app2] = varCE(X_lam, G_nu, theta_fit2, W_fit2);

subplot(2, 2, 1)
hold on
plot(theo_var./theo_mean, 'k');
plot(est_var1./est_mean1, 'r')
ylabel('Fano Factor')
title('filtering')
hold off

subplot(2, 2, 2)
hold on
plot(theo_var./theo_mean, 'k');
plot(est_var2./est_mean2, 'r')
title('smoothing')
hold off

subplot(2, 2, 3)
hold on
plot(var_rate_exact1)
plot(var_rate_app1)
ylabel('varCE')
% ylim([0 0.15])
legend('exact', 'app')
hold off

subplot(2, 2, 4)
hold on
plot(var_rate_exact2)
plot(var_rate_app2)
% ylim([0 0.15])
legend('exact', 'app')
hold off
