rng(123)
addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
T = 100;
dt = 0.01; % bin length (s)
n = 1; % number of independent observations
Q_true = diag([1e-6 1e-4]);

X_lam = normrnd(1,.5,[round(T/dt),1]);
G_nu = normrnd(1,.5,[round(T/dt),1]);

%
beta_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(1, 1))));
gamma_true = ones(1, round(T/dt))' + ...
    detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(2, 2))));

lam_true = exp(X_lam.*beta_true);
nu_true = exp(G_nu.*gamma_true);

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), n);
end

theta_true = [beta_true gamma_true];
%%
theta0 = [log(mean(spk_vec(1:100))); 0];

nQ = 10;
QLB = 1e-8;
QUB = 1e-3;

Qvec = logspace(log10(QLB), log10(QUB), nQ);
llhdmesh = zeros(nQ, nQ);

for j = 1:nQ
    for k = 1:nQ
        fprintf('Qbeta0 %02i/%02i... Qwtlong %02i/%02i...', j, nQ, k, nQ)
        [theta_fit,W_fit, lam_pred, nu_pred, Z_pred] =...
            ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
            eye(2),eye(2),diag([Qvec(j) Qvec(k)]));
        llhd_pred = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log(Z_pred));
        fprintf('llhd %.02f... \n', llhd_pred)
        llhdmesh(j ,k) = llhd_pred;
        
    end
end

% [qlam_indx, qnu_indx] = find(llhdmesh == max(max(llhdmesh)));
% Qopt = [Qvec(qlam_indx) 0; 0 Qvec(qnu_indx)];

Q0 = [QLB QLB];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 100;
MaxIter = 25;

f = @(Q) helper_2d(Q, theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],[QLB QLB],[QUB QUB], [], options);

[theta_Qopt1,W_Qopt1, ~, ~, ~] =...
    ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([Qopt(1) Qopt(2)]));

[theta_Qopt2,W_Qopt2] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    eye(2),eye(2),diag([Qopt(1) Qopt(2)]));


llhdPlot_heat = figure;
set(llhdPlot_heat,'color','w');
hold on
xlabel('log_{10}(Q_{\nu})'); ylabel('log_{10}(Q_{\lambda})');
colormap(gray(256));
colorbar;
imagesc(log10(Qvec), log10(Qvec), llhdmesh);
xlim([min(log10(Qvec))-.5, max(log10(Qvec))+.5]);
ylim([min(log10(Qvec))-.5, max(log10(Qvec))+.5]);
yline(log10(Q_true(1, 1)), 'r--', 'LineWidth', 2);
xline(log10(Q_true(2, 2)), 'r--', 'LineWidth', 2);

plot(log10(Qopt(2)), log10(Qopt(1)), 'o', 'Color', 'b',...
    'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)

saveas(llhdPlot_heat, 'llhdPlot_heat.png')
%%


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

saveas(theta, 'theta_Qtune.png')

%%








