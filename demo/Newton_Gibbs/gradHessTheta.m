function GradHess = gradHessTheta(vecTheta, X_lam,G_nu, theta0_tmp, W0_tmp,...
    F, Q_tmp, spk_vec, varargin)

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

Theta = reshape(vecTheta, [], T);
beta = Theta(1:size(X_lam, 2),:);
gam = Theta((size(X_lam, 2)+1):end,:);
maxSum = 10*max(spk_vec(:));

hessup = repmat(eye(size(F)), 1, 1, T-1);
hessub = repmat(eye(size(F)), 1, 1, T-1);
hessmed = repmat(zeros(size(Theta, 1)),1,1,T);
gradMatrix = 0*Theta;

for t = 1:T
    lam = exp(X_lam(t,:)*beta(:,t)) ;
    nu = exp(G_nu(t,:)*gam(:,t));
    [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, ~] = ...
        CMPmoment(lam, nu, maxSum);
    
    scoreTmp = [(spk_vec(t) - mean_Y)*X_lam(t,:)';...
        nu*(-gammaln(spk_vec(t) + 1) + mean_logYfac)*G_nu(t, :)'];
    
    hess1 = -var_Y*X_lam(t,:)'*X_lam(t,:);
    hess2 = nu*cov_Y_logYfac*X_lam(t,:)'*G_nu(t, :);
    hess3 = hess2';
    hess4 = -nu*(nu*var_logYfac)*G_nu(t, :)'*G_nu(t, :); % scoring
    %     hess4 = -nu*(nu*var_logYfac - mean_logYfac +...
    %         gammaln(spk_vec(t) + 1))*G_nu(t, :)'*G_nu(t, :); % exact
    
    hess = [hess1, hess2; hess3, hess4];
    
    if(t == 1)
        Fnext = F^(h(t));
        Qnext = 0;
        for l = 1:h(t); Qnext = Qnext + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        gradMatrix(:,t) = scoreTmp - W0_tmp\(Theta(:,1) - theta0_tmp)+...
            Fnext'*(Qnext\(Theta(:,2) - Fnext*Theta(:,1)));
        hessmed(:,:,t) = hess - inv(W0_tmp) - Fnext'*(Qnext\Fnext);
        
        hessup(:,:,t) = (Qnext\Fnext)';
        hessub(:,:,t) = Qnext\Fnext;
        
    elseif(t == T)
        Fthis = F^(h(t-1));
        Qthis = 0;
        for l = 1:h(t-1);Qthis = Qthis + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        
        gradMatrix(:,t) = scoreTmp - Qthis\(Theta(:,T) - Fthis*Theta(:,T-1));
        hessmed(:,:,t) = hess - inv(Qthis);
    else
        Fthis = F^(h(t-1));
        Fnext = F^(h(t));
        Qthis = 0;
        Qnext = 0;
        for l = 1:h(t-1);Qthis = Qthis + (F^(l-1))*Q_tmp*(F^(l-1))';end
        for l = 1:h(t);Qnext = Qnext + (F^(l-1))*Q_tmp*(F^(l-1))';end
        
        gradMatrix(:,t) = scoreTmp-Qthis\(Theta(:,t) -...
            Fthis*Theta(:,(t-1)))+Fnext'*(Qnext\(Theta(:,(t+1)) - Fnext*Theta(:,t)));
        hessmed(:,:,t) = hess - inv(Qthis) - Fnext'*(Qnext\Fnext);
        
        hessup(:,:,t) = (Qnext\Fnext)';
        hessub(:,:,t) = Qnext\Fnext;
    end
    
end

GradHess{1} = gradMatrix(:);
GradHess{2} = blktridiag(hessmed,hessub,hessup);


end