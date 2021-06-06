addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(123)
T = 100;
dt = 0.01; % bin length (s)
N = 1; % number of independent observations
Q_true = diag([1e-4 1e-5]);


% X_lam = ones(T/dt, 1);
% G_nu = ones(T/dt, 1);
X_lam = normrnd(1,.5,[round(T/dt),1]);
G_nu = normrnd(1,.5,[round(T/dt),1]);

%
beta_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(1, 1))));
gamma_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(2, 2))));

lam_true = exp(X_lam.*beta_true);
nu_true = exp(G_nu.*gamma_true);
spk_vec = com_rnd(lam_true, nu_true);

theta_true = [beta_true gamma_true];

%% select Q
windType = 'center';
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
windSize = 10;

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0_tmp,F,diag([1e-4 1e-4]), windSize, windType);

theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);

nQ = 5;
QLB = 1e-8;
QUB = 1e-3;
Qvec = logspace(log10(QLB), log10(QUB), nQ);
llhdmesh_pred = zeros(nQ, nQ);

for j = 1:nQ
    for k = 1:nQ
        fprintf('Qbeta0 %02i/%02i... Qwtlong %02i/%02i...', j, nQ, k, nQ)
        
        [~, ~, lam_pred, nu_pred, log_Zvec_pred] =...
            ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
            W0,eye(2),diag([Qvec(j) Qvec(k)]), windSize, windType);
        
        llhd_pred = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log_Zvec_pred);
        
        fprintf('llhd %.02f... \n', llhd_pred)
        llhdmesh_pred(j ,k) = llhd_pred;
    end
end


llhdPlot_heat = figure;
set(llhdPlot_heat,'color','w');
hold on
xlabel('log_{10}(Q_{\nu})'); ylabel('log_{10}(Q_{\lambda})');
colormap(gray(256));
colorbar;
imagesc(log10(Qvec), log10(Qvec), llhdmesh_pred);


%%
Q0 = [QLB QLB];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 100;
MaxIter = 25;

f = @(Q) helper_window(Q, theta0, spk_vec,X_lam,G_nu,W0,F,windSize, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],[QLB QLB],[QUB QUB], [], options);

%%
[theta_Qopt1,W_Qopt1] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),Q_true, windSize, windType);

[theta_Qopt2,W_Qopt2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),diag([Qopt(1) Qopt(2)]), windSize, windType);

theta = figure;
subplot(2, 1, 1)
hold on
l1 = plot(theta_true(:, 1), 'k', 'LineWidth', 2);
l2 = plot(theta_Qopt1(1, :), 'b', 'LineWidth', 1);
l3 = plot(theta_Qopt2(1, :), 'r', 'LineWidth', 1);
plot(theta_Qopt1(1, :) + sqrt(squeeze(W_Qopt1(1, 1, :)))', 'b:', 'LineWidth', 1)
plot(theta_Qopt1(1, :) - sqrt(squeeze(W_Qopt1(1, 1, :)))', 'b:', 'LineWidth', 1)
plot(theta_Qopt2(1, :) + sqrt(squeeze(W_Qopt2(1, 1, :)))', 'r:', 'LineWidth', 1)
plot(theta_Qopt2(1, :) - sqrt(squeeze(W_Qopt2(1, 1, :)))', 'r:', 'LineWidth', 1)
legend([l1 l2 l3], 'true', 'Q-true-smooth', 'Q-tune-smooth', 'Location','northwest');
title('\theta for \lambda')
xlabel('step')
xlim([100, round(T/dt)])
hold off

subplot(2, 1, 2)
hold on
l1 = plot(theta_true(:, 2), 'k', 'LineWidth', 2);
l2 = plot(theta_Qopt1(2, :), 'b', 'LineWidth', 1);
l3 = plot(theta_Qopt2(2, :), 'r', 'LineWidth', 1);
plot(theta_Qopt1(2, :) + sqrt(squeeze(W_Qopt1(2, 2, :)))', 'b:', 'LineWidth', 1)
plot(theta_Qopt1(2, :) - sqrt(squeeze(W_Qopt1(2, 2, :)))', 'b:', 'LineWidth', 1)
plot(theta_Qopt2(2, :) + sqrt(squeeze(W_Qopt2(2, 2, :)))', 'r:', 'LineWidth', 1)
plot(theta_Qopt2(2, :) - sqrt(squeeze(W_Qopt2(2, 2, :)))', 'r:', 'LineWidth', 1)
legend([l1 l2 l3], 'true', 'Q-true-smooth', 'Q-tune-smooth', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
xlim([100, round(T/dt)])
hold off






