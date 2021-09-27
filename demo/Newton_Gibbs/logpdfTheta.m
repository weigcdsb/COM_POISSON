function [lpdf, glpdf] = logpdfTheta(vecTheta, X_lam,G_nu, theta0_tmp, W0_tmp,...
    F, Q_tmp, spk_vec)

T = length(spk_vec);
Theta = reshape(vecTheta, [], T);
logNPrior =  -1/2*(Theta(:,1) - theta0_tmp)'*inv(W0_tmp)*(Theta(:,1) - theta0_tmp) -...
    1/2*trace((Theta(:,2:end) - Theta(:,1:(end-1)))'*inv(Q_tmp)*...
    (Theta(:,2:end) - Theta(:,1:(end-1))));


beta = Theta(1:size(X_lam, 2),:);
gam = Theta((size(X_lam, 2)+1):end,:);
maxSum = 10*max(spk_vec(:));

SCORE = 0*Theta;
llhd = 0;

for t = 1:T
    lam = exp(X_lam(t,:)*beta(:,t)) ;
    nu = exp(G_nu(t,:)*gam(:,t));
    [mean_Y, ~, mean_logYfac, ~, ~, log_Z] = ...
        CMPmoment(lam, nu, maxSum);
    llhd = llhd + spk_vec(t)*log(lam+(lam==0)) - nu*gammaln(spk_vec(t) + 1) - log_Z;
    
    
    SCORE(:,t) = [(spk_vec(t) - mean_Y)*X_lam(t,:)';...
        nu*(-gammaln(spk_vec(t) + 1) + mean_logYfac)*G_nu(t, :)'];
    
end

gradMat = SCORE+ [-W0_tmp\(Theta(:,1) - theta0_tmp)+...
    F'*(Q_tmp\(Theta(:,2) - F*Theta(:,1))),...
    -Q_tmp\(Theta(:,2:(T-1)) - F*Theta(:,1:(T-2)))+...
    F'*(Q_tmp\(Theta(:,3:T) - F*Theta(:,2:(T-1)))),...
    -Q_tmp\(Theta(:,T) - F*Theta(:,T-1))];
glpdf = gradMat(:)';
lpdf = llhd + logNPrior;

% out{1} = lpdf;
% out{2} = glpdf';



end