function Grad = gradTheta(vecTheta, X_lam,G_nu, theta0_tmp, W0_tmp,...
    F, Q_tmp, spk_vec)

T = length(spk_vec);
Theta = reshape(vecTheta, [], T);
beta = Theta(1:size(X_lam, 2),:);
gam = Theta((size(X_lam, 2)+1):end,:);
maxSum = 10*max(spk_vec(:));

SCORE = 0*Theta;
for t = 1:T
    lam = exp(X_lam(t,:)*beta(:,t)) ;
    nu = exp(G_nu(t,:)*gam(:,t));
    [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, ~] = ...
        CMPmoment(lam, nu, maxSum);
    
    SCORE(:,t) = [(spk_vec(t) - mean_Y)*X_lam(t,:)';...
        nu*(-gammaln(spk_vec(t) + 1) + mean_logYfac)*G_nu(t, :)'];
    
end

gradMat = SCORE+ [-W0_tmp\(Theta(:,1) - theta0_tmp)+...
    F'*(Q_tmp\(Theta(:,2) - F*Theta(:,1))),...
    -Q_tmp\(Theta(:,2:(T-1)) - F*Theta(:,1:(T-2)))+...
    F'*(Q_tmp\(Theta(:,3:T) - F*Theta(:,2:(T-1)))),...
    -Q_tmp\(Theta(:,T) - F*Theta(:,T-1))];
Grad = gradMat(:);

end