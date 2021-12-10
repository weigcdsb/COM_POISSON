function GradHess = gradHessTheta_CMP_fixNu(vecTheta, X_lam, nu, theta0_tmp,...
    W0_tmp,F, Q_tmp, spk_vec, varargin)

% to debug
% vecTheta = theta_tmp(:);
% nu = nu_trace(1);
% theta0_tmp = theta0;
% W0_tmp = 1e-2;
% F = 1;
% Q_tmp = 1e-3;


T = length(spk_vec);
maxSum = 10*max(spk_vec(:));
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

hessup = repmat(eye(size(F)), 1, 1, T-1);
hessub = repmat(eye(size(F)), 1, 1, T-1);
hessmed = repmat(zeros(size(theta, 1)),1,1,T);
gradMatrix = 0*theta;
for t = 1:T
    lam = exp(X_lam(t,:)*theta(:,t));
    [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, ~] = ...
        CMPmoment(lam, nu, maxSum);
    
    scoreTmp = (spk_vec(t) - mean_Y)*X_lam(t,:)';
    hess = -var_Y*X_lam(t,:)'*X_lam(t,:);
    
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