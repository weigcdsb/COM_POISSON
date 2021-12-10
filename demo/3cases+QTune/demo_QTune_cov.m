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

%%
windType = 'forward';
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
    W0_tmp,F,diag([1e-4 1e-4]), 1, windType);

theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-3*ones(1, min(2, size(X_lam, 2))+ min(2, size(G_nu, 2))) 0];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 200;
MaxIter = 100;


f = @(Q) helper_window_v3(Q, theta0, spk_vec,X_lam,G_nu,W0,F,1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],[QLB*ones(1, length(Q0)-1), -QUB],...
    [QUB*ones(1, length(Q0)-1), QUB], [], options);

%% fit
if(size(X_lam, 2) >= 2)
    Q_lam = [Qopt(1) Qopt(2)*ones(1, size(X_lam, 2)-1)];
    if(size(G_nu, 2) >= 2)
        Q_nu = [Qopt(3) Qopt(4)*ones(1, size(G_nu, 2) - 1)];
    else
        Q_nu = Qopt(3);
    end
else
    Q_lam = Qopt(1);
    if(size(G_nu, 2) >= 2)
        Q_nu = [Qopt(2) Qopt(3)*ones(1, size(G_nu, 2) - 1)];
    else
        Q_nu = Qopt(2);
    end
end

Qoptmatrix = diag([Q_lam Q_nu]);
Qoptmatrix(1, length(Q_lam)+1) = Qopt(end);
Qoptmatrix(length(Q_lam)+1, 1) = Qopt(end);


[theta_Qopt1,W_Qopt1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),Qoptmatrix, 1, windType);

[theta_Qopt2,W_Qopt2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),Qoptmatrix, 1, windType);

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
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
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
legend([l1 l2 l3], 'true', 'filtering', 'smoothing', 'Location','northwest');
title('\theta for \nu')
xlabel('step')
xlim([100, round(T/dt)])
hold off

save('C:\Users\gaw19004\Desktop\COM_POI_data\Qtune_cov.mat')
saveas(theta, 'theta_Qtune_cov.png')
