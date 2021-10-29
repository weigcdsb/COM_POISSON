function GradHess = gradHessTheta_Poisson(vecTheta, X_lam, theta0_tmp,...
    W0_tmp,F, Q_tmp, spk_vec, varargin)

% to debug
% vecTheta = theta_fit_tmp(:);
% X_lam = Xb;
% theta0_tmp = theta04;
% W0_tmp = W04;
% F = eye(length(theta04));
% Q_tmp = Qoptmatrix4;
% spk_vec = spk_vec;


T = length(spk_vec);
obsIdxAll = 1:T;

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'obsIdx'}
                obsIdxAll = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

h = diff(obsIdxAll);

theta = reshape(vecTheta, [], T);

% hessup = repmat((Q_tmp\F)', 1, 1, T-1);
% hessub = repmat(Q_tmp\F, 1, 1, T-1);
hessup = repmat(eye(size(F)), 1, 1, T-1);
hessub = repmat(eye(size(F)), 1, 1, T-1);
hessmed = repmat(zeros(size(theta, 1)),1,1,T);
gradMatrix = 0*theta;

for t = 1:T
    lam = exp(X_lam(t,:)*theta(:,t));
    hess = -X_lam(t,:)'*diag(lam)*X_lam(t,:);
    scoreTmp = X_lam(t,:)'*(spk_vec(t) - lam);
    
    if(t == 1)
        Fnext = F^(h(t));
        Qnext = 0;
        for l = 1:h(t); Qnext = Qnext + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        gradMatrix(:,t) = scoreTmp - W0_tmp\(theta(:,1) - theta0_tmp)+...
            Fnext'*(Qnext\(theta(:,2) - Fnext*theta(:,1)));
        hessmed(:,:,t) = hess - inv(W0_tmp) - Fnext'*(Qnext\Fnext);
        
        hessup(:,:,t) = (Qnext\Fnext)';
        hessub(:,:,t) = Qnext\Fnext;
        
    elseif(t == T)
        Fthis = F^(h(t-1));
        Qthis = 0;
        for l = 1:h(t-1);Qthis = Qthis + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        gradMatrix(:,t) = scoreTmp - Qthis\(theta(:,T) - Fthis*theta(:,T-1));
        hessmed(:,:,t) = hess - inv(Qthis);
    else
        Fthis = F^(h(t-1));
        Fnext = F^(h(t));
        Qthis = 0;
        Qnext = 0;
        for l = 1:h(t-1);Qthis = Qthis + (F^(l-1))*Q_tmp*(F^(l-1))';end
        for l = 1:h(t);Qnext = Qnext + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        gradMatrix(:,t) = scoreTmp-Qthis\(theta(:,t) -...
            Fthis*theta(:,(t-1)))+Fnext'*(Qnext\(theta(:,(t+1)) - Fnext*theta(:,t)));
        hessmed(:,:,t) = hess - inv(Qthis) - Fnext'*(Qnext\Fnext);
        
        hessup(:,:,t) = (Qnext\Fnext)';
        hessub(:,:,t) = Qnext\Fnext;
    end
end

GradHess{1} = gradMatrix(:);
GradHess{2} = blktridiag(hessmed,hessub,hessup);









end