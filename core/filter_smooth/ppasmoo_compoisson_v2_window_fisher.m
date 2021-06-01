function [theta,W,lam,nu,log_Zvec] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, N,X_lam,G_nu,W0,F,Q, windSize)

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 10*max(N(:)); % max number for sum estimation;

% Preallocate
theta   = zeros(length(theta0), n_spk);
W   = zeros([size(W0) n_spk]);
lam = n_spk*0;
nu = n_spk*0;
log_Zvec = n_spk*0;
np_lam = size(X_lam, 2);

% Initialize
theta(:,1)   = theta0;
W(:,:,1) = W0;

lam(1) = exp(X_lam(1,:)*theta0(1:np_lam));
nu(1) = exp(G_nu(1,:)*theta0((np_lam+1):end));

thetapred = theta;
Wpred = W;

% warning('Message 1.')
% Forward-Pass (Filtering)
for i=2:n_spk
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    if((i + windSize - 1) <= n_spk)
        obsIdx = i:(i+windSize-1);
    else
        obsIdx = i:n_spk;
    end
    
    lam_ext = exp(X_lam(obsIdx,:)*thetapred(1:np_lam, i));
    nu_ext = exp(G_nu(obsIdx,:)*thetapred((np_lam+1):end, i));
    lam(i) = lam_ext(1);
    nu(i) = nu_ext(1);
    
    INFO = zeros(size(W0));
    SCORE = zeros(length(theta0), 1);
    
    for k = 1:length(lam_ext)
        logcum_app = logsum_calc(lam_ext(k), nu_ext(k), maxSum);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        log_C = logcum_app(4);
        log_D = logcum_app(5);
        log_E = logcum_app(6);
        
        if(k == 1); log_Zvec(i) = log_Z; end
        
        mean_Y = exp(log_A - log_Z);
        var_Y = exp(log_B - log_Z) - mean_Y^2;
        mean_logYfac = exp(log_C - log_Z);
        var_logYfac = exp(log_D - log_Z) - mean_logYfac^2;
        cov_Y_logYfac =  exp(log_E-log_Z)-exp(log_A+log_C-2*log_Z);
        
        info1 = nCell*var_Y*X_lam(i+k-1,:)'*X_lam(i+k-1,:);
        info2 = -nCell*nu_ext(k)*cov_Y_logYfac*X_lam(i+k-1,:)'*G_nu(i+k-1, :);
        info3 = info2';
%         info4 = nu_ext(k)*(nCell*nu_ext(k)*var_logYfac - nCell*mean_logYfac +...
%             sum(gammaln(N(:, i+k-1) + 1)))*G_nu(i+k-1, :)'*G_nu(i+k-1, :);
        info4 = nu_ext(k)*(nCell*nu_ext(k)*var_logYfac)*G_nu(i+k-1, :)'*G_nu(i+k-1, :); % Fisher scoring
        
        INFO = INFO + [info1, info2; info3, info4];
        SCORE = SCORE + [(sum(N(:, i+k-1)) - nCell*mean_Y)*X_lam(i+k-1,:)';...
            nu_ext(k)*(-sum(gammaln(N(:, i+k-1) + 1)) + nCell*mean_logYfac)*G_nu(i+k-1, :)'];
    end
    
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    
    theta(:,i)  = thetapred(:,i) + W(:,:,i)*SCORE;
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix') || strcmp(msgid,'MATLAB:illConditionedMatrix')
        lastwarn('')
        return;
%         keyboard
    end
end

lastwarn('')
I = eye(length(theta0));

for i=(n_spk-2):-1:1
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(F)*(I-Q*Wi);
    Ksquig = inv(F)*Q*Wi;
    
    theta(:,i)=Fsquig*theta(:,i+1) + Ksquig*thetapred(:,i+1);
    C = W(:,:,i)*F'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    
    lam(i) = exp(X_lam(i,:)*theta(1:np_lam, i));
    nu(i) = exp(G_nu(i,:)*theta((np_lam+1):end, i));
    logcum_app = logsum_calc(lam(i), nu(i), maxSum);
    log_Zvec(i) = logcum_app(1);
end

end
