function GradHess = gradHessTheta_na(vecTheta, X_lam,G_nu, theta0_tmp, W0_tmp,...
    F, Q_tmp, spk_vec)

% to debug
% vecTheta = theta_fit_tmp(:);
% X_lam = Xb;
% G_nu = Gb_full;
% theta0_tmp = theta01;
% W0_tmp = W01;
% F = eye(length(theta01));
% Q_tmp = Qoptmatrix1;
% spk_vec = spk_vec;

T = length(spk_vec);

Theta = reshape(vecTheta, [], T);
beta = Theta(1:size(X_lam, 2),:);
gam = Theta((size(X_lam, 2)+1):end,:);
maxSum = 10*max(spk_vec(:));

hessup = repmat((Q_tmp\F)', 1, 1, T-1);
hessub = repmat(Q_tmp\F, 1, 1, T-1);
hessmed = repmat(zeros(size(Theta, 1)),1,1,T);
SCORE = 0*Theta;
nParam = size(X_lam, 2) + size(G_nu, 2);
for t = 1:T
    if ~isnan(spk_vec(t))
        lam = exp(X_lam(t,:)*beta(:,t)) ;
        nu = exp(G_nu(t,:)*gam(:,t));
        [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, ~] = ...
            CMPmoment(lam, nu, maxSum);
        
        SCORE(:,t) = [(spk_vec(t) - mean_Y)*X_lam(t,:)';...
            nu*(-gammaln(spk_vec(t) + 1) + mean_logYfac)*G_nu(t, :)'];
        
        hess1 = -var_Y*X_lam(t,:)'*X_lam(t,:);
        hess2 = nu*cov_Y_logYfac*X_lam(t,:)'*G_nu(t, :);
        hess3 = hess2';
        hess4 = -nu*(nu*var_logYfac)*G_nu(t, :)'*G_nu(t, :); % scoring
%         hess4 = -nu*(nu*var_logYfac - mean_logYfac +...
%         gammaln(spk_vec(t) + 1))*G_nu(t, :)'*G_nu(t, :); % exact
        
        hess = [hess1, hess2; hess3, hess4];
        
    else
        SCORE(:,t) = zeros(nParam, 1);
        hess = zeros(nParam);
    end
    
    if(t == 1)
        hessmed(:,:,t) = hess - inv(W0_tmp) - F'*(Q_tmp\F);
    elseif(t == T)
        hessmed(:,:,t) = hess - inv(Q_tmp);
    else
        hessmed(:,:,t) = hess - inv(Q_tmp) - F'*(Q_tmp\F);
    end
    
end

GradHess{1} = SCORE+ [-W0_tmp\(Theta(:,1) - theta0_tmp)+...
    F'*(Q_tmp\(Theta(:,2) - F*Theta(:,1))),...
    -Q_tmp\(Theta(:,2:(T-1)) - F*Theta(:,1:(T-2)))+...
    F'*(Q_tmp\(Theta(:,3:T) - F*Theta(:,2:(T-1)))),...
    -Q_tmp\(Theta(:,T) - F*Theta(:,T-1))];
GradHess{1} = GradHess{1}(:);
GradHess{2} = blktridiag(hessmed,hessub,hessup);

end