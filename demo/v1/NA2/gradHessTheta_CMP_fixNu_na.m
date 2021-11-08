function GradHess = gradHessTheta_CMP_fixNu_na(vecTheta, X_lam, nu, theta0_tmp,...
    W0_tmp,F, Q_tmp, spk_vec)

% to debug
% vecTheta = theta_tmp(:);
% X_lam = Xb;
% nu = nu_trace(1);
% theta0_tmp = theta02(1:(end-1));
% W0_tmp = W02(1:(end-1), 1:(end-1));
% F = eye(size(Xb, 2));
% Q_tmp = Qoptmatrix3;


T = length(spk_vec);
maxSum = 10*max(spk_vec(:));

theta = reshape(vecTheta, [], T);

hessup = repmat((Q_tmp\F)', 1, 1, T-1);
hessub = repmat(Q_tmp\F, 1, 1, T-1);
hessmed = repmat(zeros(size(theta, 1)),1,1,T);
SCORE = 0*theta;
nParam = size(theta, 1);

for t = 1:T
    
    if ~isnan(spk_vec(t))
        lam = exp(X_lam(t,:)*theta(:,t));
        [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, ~] = ...
        CMPmoment(lam, nu, maxSum);
        
        SCORE(:,t) = (spk_vec(t) - mean_Y)*X_lam(t,:)';
        hess = -var_Y*X_lam(t,:)'*X_lam(t,:);
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

GradHess{1} = SCORE+ [-W0_tmp\(theta(:,1) - theta0_tmp)+...
    F'*(Q_tmp\(theta(:,2) - F*theta(:,1))),...
    -Q_tmp\(theta(:,2:(T-1)) - F*theta(:,1:(T-2)))+...
    F'*(Q_tmp\(theta(:,3:T) - F*theta(:,2:(T-1)))),...
    -Q_tmp\(theta(:,T) - F*theta(:,T-1))];
GradHess{1} = GradHess{1}(:);
GradHess{2} = blktridiag(hessmed,hessub,hessup);



end