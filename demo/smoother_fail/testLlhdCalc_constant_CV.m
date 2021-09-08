function [llhd_mean, windSize] = testLlhdCalc_constant_CV(beta,gam,k,nNew)

% training dataset
X_lam = ones(k, 1);
G_nu = ones(k, 1);

theta_true = zeros(2,k);
theta_true(1,:) = beta*ones(1,k);
theta_true(2,:) = gam*ones(1,k);
lamSeq = exp(X_lam'.*theta_true(1,:));
nuSeq = exp(G_nu'.*theta_true(2,:));
spk_vec = com_rnd(lamSeq, nuSeq);

% fit models
Q = diag([1e-3 1e-3]);
windType = 'forward';
F = diag([1 1]);

theta0 = theta_true(:, 1);
W0 = eye(2)*1e-2;

theta_filt_exact =...
    ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

theta_filt =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

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
disp(windSize)
theta_filt_wind =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize, windType);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_filt(:),1e-10,1000);
theta_newton = reshape(theta_newton_vec, [], k);

% test llhd
fitted_theta = [theta_filt_exact; theta_filt;...
    theta_filt_wind; theta_newton];
mean_Y = zeros(4, k);
log_Z = zeros(4, k);
lam_all = zeros(4, k);
nu_all = zeros(4, k);
for m = 1:4
    fit = fitted_theta(((m-1)*2+1):2*m,:);
    lam_all(m,:) = exp(X_lam'.*fit(1,:));
    nu_all(m,:) = exp(G_nu'.*fit(2,:));
    for t = 1:k
        [mean_Y(m,t), ~, ~, ~, ~, log_Z(m,t)] = CMPmoment(lam_all(m,t),...
            nu_all(m,t), 1000);
    end
end

% multiple new dataset
llhd = zeros(4, nNew);
for j = 1:nNew
    spk_vec_new = com_rnd(lamSeq, nuSeq);
    for m = 1:4
        llhd(m,j) = sum(spk_vec_new.*log((lam_all(m,:)+(lam_all(m,:)==0))) -...
            nu_all(m,:).*gammaln(spk_vec_new + 1) - log_Z(m,:))/sum(spk_vec_new);
    end
end
llhd_mean = mean(llhd,2);


end