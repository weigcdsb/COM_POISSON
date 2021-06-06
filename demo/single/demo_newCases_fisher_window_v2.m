addpath(genpath('D:\GitHub\COM_POISSON'));
%%

rng(1) %rng(5)
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

windType = 'center';

%%
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
Q0 = 1e-4*ones(1, length(theta0));
DiffMinChange = QLB;
DiffMaxChange = QUB;
MaxFunEvals = 200;
MaxIter = 25;

f = @(Q) helper_window(Q, theta0, spk_vec,X_lam,G_nu,W0,F,1, windType);
% fmincon
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);

winSizeSet = [1 linspace(5, 30, 6)];
np = size(X_lam, 2) + size(G_nu, 2);
theta0_winSize = zeros(np, length(winSizeSet));
W0_winSize = zeros(np, np, length(winSizeSet));

preLL_winSize_pred = zeros(1, length(winSizeSet));
preLL_winSize_filt = zeros(1, length(winSizeSet));
preLL_winSize_smoo = zeros(1, length(winSizeSet));


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
    
    [~,~,lam_pred,nu_pred,log_Zvec_pred,...
    lam_filt,nu_filt,log_Zvec_filt,...
    lam_smoo,nu_smoo,log_Zvec_smoo] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_winSize(:, idx), spk_vec,X_lam,G_nu,...
        W0_winSize(:, :, idx),F,diag(Qopt), k, windType);
    
    if(length(log_Zvec_pred) == size(spk_vec, 2))
        preLL_winSize_pred(idx) = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk_vec + 1) - log_Zvec_pred);
        preLL_winSize_filt(idx) = sum(spk_vec.*log((lam_filt+(lam_filt==0))) -...
            nu_filt.*gammaln(spk_vec + 1) - log_Zvec_filt);
        preLL_winSize_smoo(idx) = sum(spk_vec.*log((lam_smoo+(lam_smoo==0))) -...
            nu_smoo.*gammaln(spk_vec + 1) - log_Zvec_smoo);
    else
        preLL_winSize_pred(idx) = -Inf;
        preLL_winSize_filt(idx) = -Inf;
        preLL_winSize_smoo(idx) = -Inf;
    end
    idx = idx + 1;
end

subplot(1, 3, 1)
plot(winSizeSet, preLL_winSize_pred)
title('prediction')
subplot(1, 3, 2)
plot(winSizeSet, preLL_winSize_filt)
title('filtering')
subplot(1, 3, 3)
plot(winSizeSet, preLL_winSize_smoo)
title('smoothing')


[~, winIdx] = max(preLL_winSize_filt);
optWinSize = winSizeSet(winIdx);
theta0 = theta0_winSize(:, winIdx);
W0 = W0_winSize(:, :, winIdx);

%%

[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt), optWinSize, windType);
[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt), optWinSize, windType);
[est_mean2,est_var2]=getMeanVar(exp(theta_fit2(1,:)),exp(theta_fit2(2,:)));

tiledlayout(2,3, 'TileSpacing', 'compact')
nexttile
plot(mean(spk_vec, 1));
hold on
plot(theo_mean, 'r', 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')

nexttile
line1 = plot(theo_mean, 'k');
hold on
line2 = plot(est_mean1, 'r');
line3 =plot(est_mean2, 'b');
hold off
ylabel('Mean')

nexttile
plot((theta_true(:,1)), 'k');
hold on
plot((theta_fit1(1,:)), 'r')
plot((theta_fit2(1,:)), 'b')
hold off
ylabel('beta')

nexttile
plot(theo_var./theo_mean, 'k');
hold on
plot(est_var1./est_mean1, 'r')
plot(est_var2./est_mean2, 'b')
hold off
ylim([0 4])
ylabel('Fano Factor')

nexttile
plot(theo_var, 'k');
hold on
plot(est_var1, 'r')
plot(est_var2, 'b')
hold off
ylabel('Var')

nexttile
plot((theta_true(:,2)), 'k');
hold on
plot((theta_fit1(2,:)), 'r')
plot((theta_fit2(2,:)), 'b')
hold off
ylabel('gamma')

lg = legend(nexttile(3), [line1,line2,line3], {'true', 'filtering', 'smoothing'});
lg.Location = 'northeastoutside';




