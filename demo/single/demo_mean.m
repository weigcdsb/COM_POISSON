addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
addpath(genpath('D:\GitHub\COM_POISSON'));

%%
rng(2)
T = 50;
dt = 0.01; % bin length (s)
n = 1; % number of independent observations

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

gamma_true = [repmat(.5, 1, round(T/(dt*3)))...
    repmat(-1.5, 1, round(T/(dt*3)))...
    repmat(.5, 1, T/dt - 2*round(T/(dt*3)))]';

nu_true = exp(G_nu.*gamma_true);

% set the mean to be approximate 5
target_mean = 5;
lam_true = (target_mean + (nu_true - 1)./ (2*nu_true)).^nu_true;
beta_true = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true))./X_lam;

figure(1)
subplot(1,2,1);
plot(lam_true)
subplot(1,2,2);
plot(nu_true)

spk_vec = zeros(n, T/dt);
theo_mean = zeros(T/dt, 1);
theo_var = zeros(T/dt, 1);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
    logcum_app = logsum_calc(lam_true(k), nu_true(k), 1000);
    log_Zk = logcum_app(1);
    log_Ak = logcum_app(2);
    log_Bk = logcum_app(3);
    log_Ck = logcum_app(4);
    
    theo_mean(k) = exp(log_Ak - log_Zk);
    theo_var(k) = exp(log_Bk - log_Zk) - theo_mean(k)^2;
end

figure(2)
subplot(2,1,1)
hold on
plot(mean(spk_vec, 1))
plot(theo_mean)
hold off

subplot(2,1,2)
% plot(theo_var)
plot(sqrt(theo_var))


theta_true = [beta_true gamma_true];
%%
close all
windType = 'forward';
Q = diag([1e-3 1e-3]);
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
% [theta_fit_tmp,W_fit_tmp] =...
%     ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
%     W0_tmp,F,Q, 10, windType);

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0_tmp,F,Q, 10, windType);

theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, 2);
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
f = @(Q) helper_window_v2(Q, theta0, spk_vec, X_lam, G_nu,...
    W0,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Qoptmatrix = diag(Qopt);

% window selection: if select by filtering/ prediction llhd, optSize = 1
optWinSize = 20;

[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);

% [theta_fit1,W_fit1] =...
%     ppafilt_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,Qoptmatrix, optWinSize, windType);

[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Qoptmatrix, optWinSize, windType);

% [theta_fit2,W_fit2] =...
%     ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,Qoptmatrix, optWinSize, windType);

[est_mean2,est_var2]=getMeanVar(exp(theta_fit2(1,:)),exp(theta_fit2(2,:)));

save("C:\Users\gaw19004\Desktop\COM_POI_data\meanVar_Qopt20_raw.mat")

close all
load("C:\Users\gaw19004\Desktop\COM_POI_data\meanVar_Qopt20_raw.mat")
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
ylim([min(theta_true(:, 1))-1 max(theta_true(:, 1))+1])
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
ylim([min(theta_true(:, 2))-1 max(theta_true(:, 2))+1])
title('\theta for \nu')
xlabel('step')
hold off

% saveas(theta, 'theta_mean.png')
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
ylim([0 6*max(theo_mean)])
title('mean')
xlabel('step')

subplot(2,1,2)
hold on
l1 = plot(theo_var, 'k', 'LineWidth', 2);
l2 = plot(fit1_var, 'b', 'LineWidth', 1);
l3 = plot(fit2_var, 'c', 'LineWidth', 1);
hold off
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northeast');
ylim([0 2*max(theo_var)])
title('var')
xlabel('step')

% saveas(mean_var, 'mean_var_mean.png')

%%
jumpIdx = find(abs(diff(theo_var)) > 0);
preL = 100;
posL = 300;

upIdx = (jumpIdx(1)-preL):(jumpIdx(1)+posL);
downIdx = (jumpIdx(2)-preL):(jumpIdx(2)+posL);


figure(10)
h1 = axes
hold on
l1 = plot(theo_var(upIdx), 'k', 'LineWidth', 2);
l2 = plot(fit1_var(upIdx), 'b', 'LineWidth', 2);
hold off
set(h1, 'YLim', [min(theo_var) max(theo_var)])
set(h1, 'XLim', [0 preL + posL])

h2 = axes
hold on
plot(theo_var(downIdx), 'k:', 'LineWidth', 2);
l3 = plot(fit1_var(downIdx), 'b:', 'LineWidth', 2);
hold off
set(h2, 'Ydir', 'reverse')
set(h2, 'YAxisLocation', 'Right')
set(h2, 'XLim', get(h1, 'XLim'))
set(h2, 'YLim', get(h1, 'YLim'))
set(h2, 'Color', 'None')
set(h2, 'Xtick', [])

title('filtering')
legend([l1 l2 l3], 'true', 'upward switch', 'downward switch (inverted)', 'Location','northwest');


figure(11)
h1 = axes
hold on
l1 = plot(theo_var(upIdx), 'k', 'LineWidth', 2);
l2 = plot(fit2_var(upIdx), 'c', 'LineWidth', 2);
hold off
set(h1, 'YLim', [min(theo_var) max(theo_var)])
set(h1, 'XLim', [0 preL + posL])

h2 = axes
hold on
% plot(theo_var(downIdx), 'k:', 'LineWidth', 2);
l3 = plot(fit2_var(downIdx), 'c:', 'LineWidth', 2);
hold off
set(h2, 'Ydir', 'reverse')
set(h2, 'YAxisLocation', 'Right')
set(h2, 'XLim', get(h1, 'XLim'))
set(h2, 'YLim', get(h1, 'YLim'))
set(h2, 'Color', 'None')
set(h2, 'Xtick', [])

title('smoothing')
legend([l1 l2 l3], 'true', 'upward switch', 'downward switch (inverted)', 'Location','northwest');



