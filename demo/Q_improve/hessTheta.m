function Hess = hessTheta(vecTheta, X_lam,G_nu, W0_tmp,...
    F, Q_tmp, spk_vec)
% this is only for single observation at each step

T = length(spk_vec);
Theta = reshape(vecTheta, [], T);
beta = Theta(1:size(X_lam, 2),:);
gam = Theta((size(X_lam, 2)+1):end,:);
maxSum = 10*max(spk_vec(:));

hessup = repmat((Q_tmp\F)', 1, 1, T-1);
hessub = repmat(Q_tmp\F, 1, 1, T-1);
hessmed = repmat(zeros(size(Theta, 1)),1,1,T);
for t= 1:T
    lam = exp(X_lam(t,:)*beta(:,t)) ;
    nu = exp(G_nu(t,:)*gam(:,t));
    logcum_app = logsum_calc(lam, nu, maxSum);
    
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    log_C = logcum_app(4);
    log_D = logcum_app(5);
    log_E = logcum_app(6);
    
    mean_Y = exp(log_A - log_Z);
    var_Y = exp(log_B - log_Z) - mean_Y^2;
    mean_logYfac = exp(log_C - log_Z);
    var_logYfac = exp(log_D - log_Z) - mean_logYfac^2;
    cov_Y_logYfac =  exp(log_E-log_Z)-exp(log_A+log_C-2*log_Z);
    
    hess1 = -var_Y*X_lam(t,:)'*X_lam(t,:);
    hess2 = nu*cov_Y_logYfac*X_lam(t,:)'*G_nu(t, :);
    hess3 = hess2';
    hess4 = -nu*(nu*var_logYfac)*G_nu(t, :)'*G_nu(t, :);
    hess = [hess1, hess2; hess3, hess4];
    
    if(t == 1)
        hessmed(:,:,t) = hess - inv(W0_tmp) - F'*(Q_tmp\F);
    elseif(t == T)
        hessmed(:,:,t) = hess - inv(Q_tmp);
    else
        hessmed(:,:,t) = hess - inv(Q_tmp) - F'*(Q_tmp\F);
    end
    
end

Hess = blktridiag(hessmed,hessub,hessup);



end