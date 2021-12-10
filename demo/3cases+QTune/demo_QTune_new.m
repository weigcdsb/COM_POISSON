addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(123)
T = 100;
dt = 0.01; % bin length (s)
N = 1; % number of independent observations
% Q_true = diag([1e-7 1e-4]);
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

for k = 1:(T/dt)
    spk_vec(:, k) = com_rnd(lam_true(k), nu_true(k), N);
end

theta_true = [beta_true gamma_true];

%% select Q
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

nQ = 10;
QLB = 1e-8;
QUB = 1e-3;
Qvec = logspace(log10(QLB), log10(QUB), nQ);
llhdmesh_pred = zeros(nQ, nQ);

for j = 1:nQ
    for k = 1:nQ
        fprintf('Qbeta0 %02i/%02i... Qwtlong %02i/%02i...', j, nQ, k, nQ)
        
        [~, ~, lam_pred, nu_pred, log_Zvec_pred,...
            lam_filt, nu_filt, log_Zvec_filt] =...
            ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
            W0,eye(2),diag([Qvec(j) Qvec(k)]), 1, windType);
        
        llhd_pred = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log_Zvec_pred);
        
        fprintf('llhd %.02f... \n', llhd_pred)
        llhdmesh_pred(j ,k) = llhd_pred;
    end
end

Q0 = [QLB QLB];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 100;
MaxIter = 25;

f = @(Q) helper_window(Q, theta0, spk_vec,X_lam,G_nu,W0,F,1,windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],[QLB QLB],[QUB QUB], [], options);


llhdPlot_heat = figure;
set(llhdPlot_heat,'color','w');
hold on
xlabel('log_{10}(Q_{\nu})'); ylabel('log_{10}(Q_{\lambda})');
colormap(gray(256));
colorbar;
imagesc(log10(Qvec), log10(Qvec), llhdmesh_pred);
xlim([min(log10(Qvec))-.5, max(log10(Qvec))+.5]);
ylim([min(log10(Qvec))-.5, max(log10(Qvec))+.5]);
yline(log10(Q_true(1, 1)), 'r--', 'LineWidth', 2);
xline(log10(Q_true(2, 2)), 'r--', 'LineWidth', 2);

plot(log10(Qopt(2)), log10(Qopt(1)), 'o', 'Color', 'b',...
    'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)

saveas(llhdPlot_heat, 'llhdPlot_heat.png')

%% select window size
winSizeSet = [1 linspace(5, 30, 6)];
np = size(X_lam, 2) + size(G_nu, 2);
theta0_winSize = zeros(np, length(winSizeSet));
W0_winSize = zeros(np, np, length(winSizeSet));

preLL_winSize = zeros(1, length(winSizeSet));
filtLL_winSize = zeros(1, length(winSizeSet));
smooLL_winSize = zeros(1, length(winSizeSet));

idx = 1;
for k = winSizeSet
    max_init = 100;
    spk_tmp = spk_vec(1:max_init);
    theta0_tmp = [log(mean(spk_tmp)); 0];
    W0_tmp = diag([1 1]);
    F = diag([1 1]);
    [theta_fit_tmp,W_fit_tmp] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
        W0_tmp,F,diag(Qopt), k, windType);
    
    theta0_winSize(:, idx) = theta_fit_tmp(:, 1);
    W0_winSize(:, :, idx) = W_fit_tmp(:, :, 1);
    
    [~, ~, lam_pred, nu_pred, log_Zvec_pred,...
        lam_filt, nu_filt, log_Zvec_filt,...
        lam_smoo, nu_smoo, log_Zvec_smoo] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_winSize(:, idx), spk_vec,X_lam,G_nu,...
        W0_winSize(:, :, idx),F, diag(Qopt), k, windType);
    
    if(length(log_Zvec_pred) == size(spk_vec, 2))
        preLL_winSize(idx) = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log_Zvec_pred);
        filtLL_winSize(idx) = sum(spk_vec.*log((lam_filt+(lam_filt==0))) -...
            nu_filt.*gammaln(spk_vec + 1) - log_Zvec_filt);
        smooLL_winSize(idx) = sum(spk_vec.*log((lam_smoo+(lam_smoo==0))) -...
            nu_smoo.*gammaln(spk_vec + 1) - log_Zvec_smoo);
    else
        preLL_winSize(idx) = -Inf;
        filtLL_winSize(idx) = -Inf;
        smooLL_winSize(idx) = -Inf;
    end
    idx = idx + 1;
end

subplot(1, 3, 1)
plot(winSizeSet, preLL_winSize)
title('prediction')
subplot(1, 3, 2)
plot(winSizeSet, filtLL_winSize)
title('filtering')
subplot(1, 3, 3)
plot(winSizeSet, smooLL_winSize)
title('smoothing')

% [~, winIdx] = max(preLL_winSize);
[~, winIdx] = max(filtLL_winSize);
% [~, winIdx] = max(smooLL_winSize);

optWinSize = winSizeSet(winIdx);
theta0 = theta0_winSize(:, winIdx);
W0 = W0_winSize(:, :, winIdx);

%% fit

[theta_Qopt1,W_Qopt1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),diag([Qopt(1) Qopt(2)]), optWinSize, windType);

[theta_Qopt2,W_Qopt2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,eye(2),diag([Qopt(1) Qopt(2)]), optWinSize, windType);

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

%% multi
rng(12)

nQ = 4;
nSeed = 5;
Q_true_seq = logspace(-7, -4, nQ);
% QLB = 1e-8;
% QUB = 1e-3;

Qlam = zeros(nQ, nQ, nSeed);
Qnu = zeros(nQ, nQ, nSeed);
N = 1;

for n = 1:nQ
    for m = 1:nQ
        for k = 1:nSeed
            fprintf('n= %i...', n)
            fprintf('m= %i...', m)
            fprintf('k= %i...', k)
            fprintf('\n')
            
            Q_true = diag([Q_true_seq(n) Q_true_seq(m)]);
            beta_true = ones(1, round(T/dt))' + ...
                detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(1, 1))));
            gamma_true = ones(1, round(T/dt))' + ...
                detrend(cumsum(randn(round(T/dt),1)*sqrt(Q_true(2, 2))));
            
            lam_true = exp(X_lam.*beta_true);
            nu_true = exp(G_nu.*gamma_true);
            
            for l = 1:(T/dt)
                spk_vec(:, l) = com_rnd(lam_true(l), nu_true(l), N);
            end
            
            max_init = 100;
            spk_tmp = spk_vec(1:max_init);
            theta0_tmp = [log(mean(spk_tmp)); 0];
            W0_tmp = diag([1 1]);
            F = diag([1 1]);
            [theta_fit_tmp,W_fit_tmp] =...
                ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
                W0_tmp,F,diag([1e-4 1e-4]), 1);
            
            theta0 = theta_fit_tmp(:, 1);
            W0 = W_fit_tmp(:, :, 1);
            
            f = @(Q) helper_window(Q, theta0, spk_vec,X_lam,G_nu,W0,F,1, windType);
            Qopt = fmincon(f,Q0,[],[],[],[],[QLB QLB],[QUB QUB], [], options);
            
            Qlam(n, m, k) = Qopt(1);
            Qnu(n, m, k) = Qopt(2);
            
        end
    end
end

%% plot

LCol = [0.9290, 0.6940, 0.1250]; % yellow
UCol = [0.25, 0.25, 0.25];
colGrad = [linspace(LCol(1),UCol(1),nQ)',...
    linspace(LCol(2),UCol(2),nQ)', linspace(LCol(3),UCol(3),nQ)'];

Q_lam = figure;
hold on
for i = 1:nQ
    for j = 1:nSeed
        jitters = normrnd(0, 0.1, 1, nQ);
        plot(log10(Q_true_seq) + jitters, log10(Qlam(:, i, j)), 'o',...
            'Color', colGrad(i,:),'LineWidth', 2, 'markerfacecolor',colGrad(i,:))
    end
end
plot([-10, -1], [-10, -1], 'r--', 'LineWidth', 2)
xlim([-10 -1])
set(gca,'FontSize',15, 'LineWidth', 1.5,'TickDir','out')
box off
hold off

set(Q_lam,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(Q_lam, 'Q_lam.svg')
saveas(Q_lam, 'Q_lam.png')

%%

for i = 1:nSeed
    Qnu(:, :, i) =  Qnu(:, :, i)';
end

LCol = [0.7490, 0.5608, 0.8902]; % purple
UCol = [0.25, 0.25, 0.25];
colGrad = [linspace(LCol(1),UCol(1),nQ)',...
    linspace(LCol(2),UCol(2),nQ)', linspace(LCol(3),UCol(3),nQ)'];

Q_nu = figure;
hold on
for i = 1:nQ
    for j = 1:nSeed
        jitters = normrnd(0, 0.1, 1, nQ);
        plot(log10(Q_true_seq) + jitters, log10(Qnu(:, i, j)), 'o',...
            'Color', colGrad(i,:),'LineWidth', 2, 'markerfacecolor',colGrad(i,:))
    end
end
plot([-10, -1], [-10, -1], 'r--', 'LineWidth', 2)
xlim([-10 -1])
set(gca,'FontSize',15, 'LineWidth', 1.5,'TickDir','out')
box off
hold off

set(Q_nu,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(Q_nu, 'Q_nu.svg')
saveas(Q_nu, 'Q_nu.png')


save('C:\Users\gaw19004\Desktop\COM_POI_data\Qtune_new.mat')





