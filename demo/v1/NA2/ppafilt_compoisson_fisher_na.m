function [theta,W,lam_pred,nu_pred,log_Zvec_pred,...
    lam_filt,nu_filt,log_Zvec_filt] =...
    ppafilt_compoisson_fisher_na(theta0, N,X_lam,G_nu,W0,F,Q)

n_spk = size(N, 2);
% nCell = size(N, 1);
maxSum = 10*max(N(:)); % max number for sum estimation;


% Preallocate
theta   = zeros(length(theta0), n_spk);
W   = zeros([size(W0) n_spk]);
lam_pred = n_spk*0;
nu_pred = n_spk*0;
log_Zvec_pred = zeros(1,n_spk)*nan;
np_lam = size(X_lam, 2);

% Initialize
theta(:,1)   = theta0;
W(:,:,1) = W0;

lam_pred(1) = exp(X_lam(1,:)*theta0(1:np_lam));
nu_pred(1) = exp(G_nu(1,:)*theta0((np_lam+1):end));
logcum_app = logsum_calc(lam_pred(1), nu_pred(1), maxSum);
log_Zvec_pred(1) = logcum_app(1);

lam_filt = lam_pred;
nu_filt = nu_pred;
log_Zvec_filt = log_Zvec_pred;

thetapred = theta;
Wpred = W;

% warning('Message 1.')
% Forward-Pass (Filtering)
for i=2:n_spk
    
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    lam_pred(i) = exp(X_lam(i,:)*thetapred(1:np_lam, i));
    nu_pred(i) = exp(G_nu(i,:)*thetapred((np_lam+1):end, i));
    
    INFO = zeros(size(W0));
    SCORE = zeros(length(theta0), 1);
    
    if ~isnan(N(:, i))
        [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, log_Zvec_pred(i)] = ...
            CMPmoment(lam_pred(i), nu_pred(i), maxSum);
        
        info1 = var_Y*X_lam(i,:)'*X_lam(i,:);
        info2 = -nu_pred(i)*cov_Y_logYfac*X_lam(i,:)'*G_nu(i, :);
        info3 = info2';
        info4 = nu_pred(i)*(nu_pred(i)*var_logYfac)*G_nu(i, :)'*G_nu(i, :); % Fisher scoring
        
        INFO = INFO + [info1, info2; info3, info4];
        SCORE = SCORE + [(N(:, i) - mean_Y)*X_lam(i,:)';...
            nu_pred(i)*(-gammaln(N(:, i) + 1) + mean_logYfac)*G_nu(i, :)'];
    end
    
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    theta(:,i)  = thetapred(:,i) + W(:,:,i)*SCORE;
    
    if ~isnan(N(:, i))
        lam_filt(i) = exp(X_lam(i,:)*theta(1:np_lam, i));
        nu_filt(i) = exp(G_nu(i,:)*theta((np_lam+1):end, i));
        logcum_app = logsum_calc(lam_filt(i), nu_filt(i), maxSum);
        log_Zvec_filt(i) = logcum_app(1);
    end
    
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix') || strcmp(msgid,'MATLAB:illConditionedMatrix')
        lastwarn('')
        error('singular')
        return;
    end
end

lastwarn('')

end
