addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%% see special cases
rng(1)
k = 200;
X_lam = ones(k, 1);
G_nu = ones(k, 1);

theta_true = zeros(2,k);

% constant
theta_true(1,:) = 2*ones(1,k); % min = 0.5, max = 3
theta_true(2,:) = 1*ones(1,k); % min = -.5, max = 2.5

% linear 
% betaStart = 0; betaEnd = 2;
% gamStart = -1; gamEnd = 1;
% theta_true(1,:) = linspace(betaStart, betaEnd, k);
% theta_true(2,:) = linspace(gamStart, gamEnd, k);

lamSeq = exp(X_lam'.*theta_true(1,:));
nuSeq = exp(G_nu'.*theta_true(2,:));

spk_vec = com_rnd(lamSeq, nuSeq);
[mean_true,var_true] = getMeanVar(lamSeq, nuSeq);

subplot(1,3,1)
plot(lamSeq)
title('\lambda')
subplot(1,3,2)
plot(nuSeq)
title('\nu')
subplot(1,3,3)
hold on
plot(spk_vec)
plot(mean_true, 'LineWidth', 2)
hold off
title('spks')

%%
Q = diag([1e-3 1e-3]);
windType = 'forward';
F = diag([1 1]);

theta0 = theta_true(:, 1);
W0 = eye(2)*1e-2;

[theta_filt_exact, W_filt_exact] =...
    ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

theta_filt =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

% set arbitrary window size
% windSize = 5;
% theta_filt_wind =...
%     ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,Q, windSize, windType);

% select by forward chaining: 
% brute-force implementation: need careful coding later
windSize0 = 1;
searchStep = 2;
windUB = 150;
windSet = windSize0:searchStep:windUB;
nSearchMax = length(windSet);

llhd_ho = [];
nDec = 0;
llhd_ho_pre = -Inf;
for w = 1:nSearchMax
    windTmp = windSet(w);
    theta_tmp = zeros(2, k-1);
    [theta_windAll_tmp, W_windAll_tmp] =...
        ppafilt_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
        W0,F,Q, windTmp, windType);
    for t = 1:(k-1)
        if (t <= windTmp)
            theta_windTmp =...
            ppafilt_compoisson_v2_window_fisher(theta0, spk_vec(1:t),X_lam(1:t),G_nu(1:t),...
            W0,F,Q, windTmp, windType, 'maxSum', 10*max(spk_vec));
        else
            theta_windTmp =...
            ppafilt_compoisson_v2_window_fisher(theta_windAll_tmp(:, t-windTmp),...
            spk_vec((t-windTmp):t),...
            X_lam((t-windTmp):t),G_nu((t-windTmp):t),...
             W_windAll_tmp(:,:,t-windTmp),F,Q, windTmp, windType, 'maxSum', 10*max(spk_vec));
        end
        theta_tmp(:,t) = theta_windTmp(:,end);
    end
    
    lam_ho = exp(theta_tmp(1,:));
    nu_ho = exp(theta_tmp(2,:));
    
    log_Z_ho = zeros(1, k-1);
    for h = 1:(k-1)
        [~, ~, ~, ~, ~, log_Z_ho(h)] = CMPmoment(lam_ho(h),nu_ho(h), 1000);
    end
    
    llhd_ho_tmp = sum(spk_vec(2:k).*log((lam_ho+(lam_ho==0))) -...
        nu_ho.*gammaln(spk_vec(2:k) + 1) - log_Z_ho)/sum(spk_vec(2:k));
    llhd_ho = [llhd_ho llhd_ho_tmp];
    if(llhd_ho_tmp < llhd_ho_pre)
        nDec = nDec + 1;
    else
        nDec = 0;
    end
    llhd_ho_pre = llhd_ho_tmp;
    
    if nDec > 2
        break
    end 
end
[~, wIdx] = max(llhd_ho);
windSize = windSet(wIdx);
theta_filt_wind =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize, windType);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_filt(:),1e-10,1000);
theta_newton = reshape(theta_newton_vec, [], k);

%%
figure;
subplot(1,2,1)
hold on
plot(theta_true(1,:))
plot(theta_filt_exact(1,:))
plot(theta_filt(1,:))
plot(theta_filt_wind(1,:))
plot(theta_newton(1,:))
hold off
title('\beta')
legend('true', 'smoother-exact', 'smoother', "window-"+windSize,'NR',...
    'Location','northwest')
subplot(1,2,2)
hold on
plot(theta_true(2,:))
plot(theta_filt_exact(2,:))
plot(theta_filt(2,:))
plot(theta_filt_wind(2,:))
plot(theta_newton(2,:))
title('\gamma')
hold off

figure;
subplot(1,2,1)
hold on
plot(exp(X_lam'.*theta_true(1,:)))
plot(exp(X_lam'.*theta_filt_exact(1,:)))
plot(exp(X_lam'.*theta_filt(1,:)))
plot(exp(X_lam'.*theta_filt_wind(1,:)))
plot(exp(X_lam'.*theta_newton(1,:)))
hold off
title('\lambda')
legend('true', 'smoother-exact', 'smoother', "window-"+windSize,'NR',...
    'Location','best')
subplot(1,2,2)
hold on
plot(exp(G_nu'.*theta_true(2,:)))
plot(exp(G_nu'.*theta_filt_exact(2,:)))
plot(exp(G_nu'.*theta_filt(2,:)))
plot(exp(G_nu'.*theta_filt_wind(2,:)))
plot(exp(G_nu'.*theta_newton(2,:)))
title('\nu')
hold off

[est_mean_filt_exact,est_var_filt_exact]=...
    getMeanVar(exp(X_lam'.*theta_filt_exact(1,:)),exp(G_nu'.*theta_filt_exact(2,:)));
[est_mean_filt,est_var_filt]=...
    getMeanVar(exp(X_lam'.*theta_filt(1,:)),exp(G_nu'.*theta_filt(2,:)));
[est_mean_filt_wind,est_var_filt_wind]=...
    getMeanVar(exp(X_lam'.*theta_filt_wind(1,:)),exp(G_nu'.*theta_filt_wind(2,:)));
[est_mean_newton,est_var_newton]=...
    getMeanVar(exp(X_lam'.*theta_newton(1,:)),exp(G_nu'.*theta_newton(2,:)));

figure;
subplot(1,2,1)
hold on
plot(mean_true)
plot(est_mean_filt_exact)
plot(est_mean_filt)
plot(est_mean_filt_wind)
plot(est_mean_newton)
hold off
title('Mean')
legend('true', 'smoother-exact', 'smoother', "window-"+windSize,'NR',...
    'Location','best')
subplot(1,2,2)
hold on
plot(var_true./ mean_true)
plot(est_var_filt_exact./ est_mean_filt_exact)
plot(est_var_filt./ est_mean_filt)
plot(est_var_filt_wind./ est_mean_filt_wind)
plot(est_var_newton./ est_mean_newton)
title('FF')
hold off



