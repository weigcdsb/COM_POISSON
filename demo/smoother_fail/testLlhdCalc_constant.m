function llhd_mean = testLlhdCalc_constant(beta,gam,k,nNew, windSize)

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