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

load('C:\Users\gaw19004\Desktop\COM_POI_data\case3.mat')

figure(1)
plot(mean(spk_vec, 1));
hold on
plot(theo_mean, 'r', 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')
ylim([round(min(spk_vec)) - 5 round(max(spk_vec)) + 5 ])

figure(2)
plotAll_filtSmoo(X_lam, G_nu, theta_true, theta_fit1, theta_fit2, W_fit1, W_fit2)

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
ylim([0 max(var_rate_exact1(100:(end - 100)))])
legend('exact', 'app')
hold off

subplot(2, 2, 4)
hold on
plot(var_rate_exact2)
plot(var_rate_app2)
ylim([0 max(var_rate_exact2(100:(end - 100)))])
legend('exact', 'app')
hold off

%% supplement: 
% (1) no Fisher + no window
% (2) no Fisher + window
% (3) Fisher + no window
% (4) Fihser + window (fitted)


% (1) no Fisher + no window
[theta_fitA1,W_fitA1] =...
    ppafilt_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, 1, windType);
[est_meanA1,est_varA1]=getMeanVar(exp(theta_fitA1(1,:)),exp(theta_fitA1(2,:)));

[theta_fitA2,W_fitA2] =...
    ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, 1, windType);
[est_meanA2,est_varA2]=getMeanVar(exp(theta_fitA2(1,:)),exp(theta_fitA2(2,:)));

figure(11)
plotAll_filtSmoo_v2(theta_true, theta_fitA1, theta_fitA2)


% (2) no Fisher + window
[theta_fitB1,W_fitB1] =...
    ppafilt_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);
[est_meanB1,est_varB1]=getMeanVar(exp(theta_fitB1(1,:)),exp(theta_fitB1(2,:)));

[theta_fitB2,W_fitB2] =...
    ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);
[est_meanB2,est_varB2]=getMeanVar(exp(theta_fitB2(1,:)),exp(theta_fitB2(2,:)));

figure(12)
plotAll_filtSmoo_v2(theta_true, theta_fitB1, theta_fitB2)

% (3) Fisher + no window
[theta_fitC1,W_fitC1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, 1, windType);
[est_meanC1,est_varC1]=getMeanVar(exp(theta_fitC1(1,:)),exp(theta_fitC1(2,:)));

[theta_fitC2,W_fitC2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, 1, windType);
[est_meanC2,est_varC2]=getMeanVar(exp(theta_fitC2(1,:)),exp(theta_fitC2(2,:)));

figure(13)
plotAll_filtSmoo_v2(theta_true, theta_fitC1, theta_fitC2)

% (4) Fisher + window
figure(14)
plotAll_filtSmoo_v2(theta_true, theta_fit1, theta_fit2)

