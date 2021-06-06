function [theta,W,lam_pred,nu_pred,log_Zvec_pred,...
    lam_filt,nu_filt,log_Zvec_filt,...
    lam_smoo,nu_smoo,log_Zvec_smoo] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, N,X_lam,G_nu,W0,F,Q, windSize, windType)

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 10*max(N(:)); % max number for sum estimation;

% Preallocate
theta   = zeros(length(theta0), n_spk);
W   = zeros([size(W0) n_spk]);
lam_pred = n_spk*0;
nu_pred = n_spk*0;
log_Zvec_pred = n_spk*0;

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

if(windType == "center")
    windSize = ceil((windSize-1)/2)*2 + 1;
%     fprintf('CAUTION: the windSize is reset as: %d', windSize);
    halfLength = (windSize-1)/2;
end

% warning('Message 1.')
% Forward-Pass (Filtering)
for i=2:n_spk
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    switch windType
        case{'forward'}
            obsIdx = i:min(n_spk, i+windSize-1);
        case{'backward'}
            obsIdx = max(1, i-windSize+1):i;
        case{'center'}
            obsIdx = max(1, i - halfLength):min(n_spk, i + halfLength);
        otherwise
            disp("please input correct window type: 'forward', 'backward' or 'center'");
            return;
    end
    
    lam_ext = exp(X_lam(obsIdx,:)*thetapred(1:np_lam, i));
    nu_ext = exp(G_nu(obsIdx,:)*thetapred((np_lam+1):end, i));
    lam_pred(i) = lam_ext(1);
    nu_pred(i) = nu_ext(1);
    
    INFO = zeros(size(W0));
    SCORE = zeros(length(theta0), 1);
    idx = 1;
    
    for k = obsIdx
        logcum_app = logsum_calc(lam_ext(idx), nu_ext(idx), maxSum);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        log_C = logcum_app(4);
        log_D = logcum_app(5);
        log_E = logcum_app(6);
        
        if(idx == 1); log_Zvec_pred(i) = log_Z; end
        
        mean_Y = exp(log_A - log_Z);
        var_Y = exp(log_B - log_Z) - mean_Y^2;
        mean_logYfac = exp(log_C - log_Z);
        var_logYfac = exp(log_D - log_Z) - mean_logYfac^2;
        cov_Y_logYfac =  exp(log_E-log_Z)-exp(log_A+log_C-2*log_Z);
        
        info1 = nCell*var_Y*X_lam(k,:)'*X_lam(k,:);
        info2 = -nCell*nu_ext(idx)*cov_Y_logYfac*X_lam(k,:)'*G_nu(k, :);
        info3 = info2';
%         info4 = nu_ext(idx)*(nCell*nu_ext(idx)*var_logYfac - nCell*mean_logYfac +...
%             sum(gammaln(N(:, k) + 1)))*G_nu(k, :)'*G_nu(k, :);
        info4 = nu_ext(idx)*(nCell*nu_ext(idx)*var_logYfac)*G_nu(k, :)'*G_nu(k, :); % Fisher scoring
        
        INFO = INFO + [info1, info2; info3, info4];
        SCORE = SCORE + [(sum(N(:, k)) - nCell*mean_Y)*X_lam(k,:)';...
            nu_ext(idx)*(-sum(gammaln(N(:, k) + 1)) + nCell*mean_logYfac)*G_nu(k, :)'];
        idx = idx + 1;
    end
    
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    
    theta(:,i)  = thetapred(:,i) + W(:,:,i)*SCORE;
    
    lam_filt(i) = exp(X_lam(i,:)*theta(1:np_lam, i));
    nu_filt(i) = exp(G_nu(i,:)*theta((np_lam+1):end, i));
    logcum_app = logsum_calc(lam_filt(i), nu_filt(i), maxSum);
    log_Zvec_filt(i) = logcum_app(1);
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix') || strcmp(msgid,'MATLAB:illConditionedMatrix')
        lastwarn('')
        return;
%         keyboard
    end
end

lastwarn('')
I = eye(length(theta0));
lam_smoo = lam_filt;
nu_smoo = nu_filt;
log_Zvec_smoo = log_Zvec_filt;

for i=(n_spk-1):-1:1
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(F)*(I-Q*Wi);
    Ksquig = inv(F)*Q*Wi;
    
    theta(:,i)=Fsquig*theta(:,i+1) + Ksquig*thetapred(:,i+1);
    C = W(:,:,i)*F'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    
    lam_smoo(i) = exp(X_lam(i,:)*theta(1:np_lam, i));
    nu_smoo(i) = exp(G_nu(i,:)*theta((np_lam+1):end, i));
    logcum_app = logsum_calc(lam_smoo(i), nu_smoo(i), maxSum);
    log_Zvec_smoo(i) = logcum_app(1);
end

end
