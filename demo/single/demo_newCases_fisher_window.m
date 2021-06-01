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

% % Case 2 -- Var decrease - constant(ish) mean (not bad)
% target_mean = 10;
% theta_true(:,2) = 5*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% % theta_true(:,1) = log(10.^nu_true); % better approximation...
% theta_true(:,1) = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true));
% Q=diag([1e-3 1e-3]);

% Case 3 -- Mean increase + Var decrease
theta_true(:,2) = 3*(t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2);
nu_true = exp(G_nu.*theta_true(:, 2));
% theta_true(:,1) = log(matchMean(exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1),nu_true));
% to run fast... use approximation again
target_mean = exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1);
theta_true(:,1) = nu_true.*log(target_mean' + (nu_true - 1)./ (2*nu_true));
Q=diag([1e-3 1e-3]);


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = com_rnd(lam_true, nu_true);


% fit... different windwo
for m = [1 5 10 100]
    
    % initialize...
    max_init = 100;
    spk_tmp = spk_vec(1:max_init);
    theta0_tmp = [log(mean(spk_tmp)); 0];
    W0_tmp = diag([1 1]);
    F = diag([1 1]);
    [theta_fit_tmp1,W_fit_tmp1] =...
        ppasmoo_compoisson_v2_window(theta0_tmp, spk_vec,X_lam,G_nu,...
        W0_tmp,F,diag([1e-4 1e-4]), m);
    [theta_fit_tmp2,W_fit_tmp2] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
        W0_tmp,F,diag([1e-4 1e-4]), m);
    
    theta01 = theta_fit_tmp1(:, 1);
    W01 = W_fit_tmp1(:, :, 1);
    
    theta02 = theta_fit_tmp2(:, 1);
    W02 = W_fit_tmp2(:, :, 1);
    
    %
    [theta_fit1,~] =...
        ppafilt_compoisson_v2_window(theta01, spk_vec,X_lam,G_nu,...
        W01,F,Q, m);
    [theta_fit2,~] =...
        ppafilt_compoisson_v2_window_fisher(theta02, spk_vec,X_lam,G_nu,...
        W02,F,Q, m);
    
    plotAll(m+100, spk_vec, X_lam, G_nu, theta_true, theta_fit1)
    plotAll(m+200, spk_vec, X_lam, G_nu, theta_true, theta_fit2)
    
end


%% turn on the optimization...
% way 1: optimize Q & window size...
% F = diag([1 1]);
% QLB = 1e-8;
% QUB = 1e-3;
% nWindLB = 1;
% nWindUB = 15;
% np = size(X_lam, 2) + size(G_nu, 2);
%
% theta0Set = zeros(np, nWindUB - nWindLB + 1);
% W0Set = zeros(np, np, nWindUB - nWindLB + 1);
%
%
% for k = nWindLB:nWindUB
%
%     max_init = 100;
%     spk_tmp = spk_vec(1:max_init);
%     theta0_tmp = [log(mean(spk_tmp)); 0];
%     W0_tmp = diag([1 1]);
%     [theta_fit_tmp,W_fit_tmp] =...
%         ppasmoo_compoisson_v2_window(theta0_tmp, spk_vec,X_lam,G_nu,...
%         W0_tmp,F,diag([1e-4 1e-4]), k);
%
%     theta0Set(:, (k -nWindLB + 1)) = theta_fit_tmp(:, 1);
%     W0Set(:, :, (k -nWindLB + 1)) = W_fit_tmp(:, :, 1);
%
% end
%
% f = @(p) helper_window_v2(p, spk_vec,X_lam,G_nu, F, theta0Set, W0Set, nWindLB);
% opts.MaxFunctionEvaluations = 500;
% p = surrogateopt(f, [QLB*ones(1, np), nWindLB], [QUB*ones(1, np), nWindUB], np+1, opts);

% wow, super inefficient...

% way2: set window size as a hyperparameter (under Q = [1e-4, 1e-4])

winSizeSet = [1 linspace(5, 30, 6)];
np = size(X_lam, 2) + size(G_nu, 2);
preLL_winSize = zeros(1, length(winSizeSet));
theta0_winSize = zeros(np, length(winSizeSet));
W0_winSize = zeros(np, np, length(winSizeSet));

idx = 1;
for k = winSizeSet
    max_init = 100;
    spk_tmp = spk_vec(1:max_init);
    theta0_tmp = [log(mean(spk_tmp)); 0];
    W0_tmp = diag([1 1]);
    F = diag([1 1]);
    [theta_fit_tmp,W_fit_tmp] =...
        ppasmoo_compoisson_v2_window_fisher(theta0_tmp, spk_vec,X_lam,G_nu,...
        W0_tmp,F,diag([1e-4 1e-4]), k);
    
    theta0_winSize(:, idx) = theta_fit_tmp(:, 1);
    W0_winSize(:, :, idx) = W_fit_tmp(:, :, 1);
    
    [~, ~, lam, nu, log_Zvec] =...
        ppafilt_compoisson_v2_window_fisher(theta0_winSize(:, idx), spk_vec,X_lam,G_nu,...
        W0_winSize(:, :, idx),F,diag([1e-4 1e-4]), k);
    
    if(length(log_Zvec) == size(spk_vec, 2))
        preLL_winSize(idx) = sum(spk_vec.*log((lam+(lam==0))) -...
            nu.*gammaln(spk_vec + 1) - log_Zvec);
    else
        preLL_winSize(idx) = -Inf;
    end
    idx = idx + 1;
end

% plot(winSizeSet, preLL_winSize)
[~, winIdx] = max(preLL_winSize);
optWinSize = winSizeSet(winIdx);
theta0 = theta0_winSize(:, winIdx);
W0 = W0_winSize(:, :, winIdx);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, length(theta0));
DiffMinChange = QLB;
DiffMaxChange = QUB;
MaxFunEvals = 200;
MaxIter = 25;

f = @(Q) helper_window(Q, theta0, spk_vec,X_lam,G_nu,W0,F,optWinSize);

% fmincon
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);


[theta_fit1,W_fit1] =...
    ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt), optWinSize);
[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt), optWinSize);
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






