function GradHess = gradHessTheta_Poisson_nan(vecTheta, X_lam, theta0_tmp,...
    W0_tmp,F, Q_tmp, spk_vec)

% to debug
% vecTheta = theta_fit_tmp(:);
% X_lam = Xb;
% theta0_tmp = theta04;
% W0_tmp = W04;
% F = eye(length(theta04));
% Q_tmp = Qoptmatrix4;
% spk_vec = spk_vec;


T = length(spk_vec);
theta = reshape(vecTheta, [], T);

hessup = repmat((Q_tmp\F)', 1, 1, T-1);
hessub = repmat(Q_tmp\F, 1, 1, T-1);
hessmed = repmat(zeros(size(theta, 1)),1,1,T);

SCORE = 0*theta;
nParam = size(theta, 1);

for t = 1:T
    
    
    if ~isnan(spk_vec(t))
        lam = exp(X_lam(t,:)*theta(:,t));
        SCORE(:,t) = X_lam(t,:)'*(spk_vec(t) - lam);
        hess = -X_lam(t,:)'*diag(lam)*X_lam(t,:);
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