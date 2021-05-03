function [theta,W] = ppasmoo_compoisson_v3(theta0, N,X_lam,G_nu,W0,F,Q)

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 10*max(N(:)); % max number for sum estimation;

% Preallocate
theta   = zeros(length(theta0), n_spk);
W   = zeros([size(W0) n_spk]);
lam = zeros(nCell, n_spk);
nu = zeros(nCell, n_spk);
np_lam = size(X_lam, 2);

% Initialize
theta(:,1)   = theta0;
W(:,:,1) = W0;

lam(:, 1) = exp(X_lam(:, :, 1) * theta0(1:np_lam));
nu(:, 1) = exp(G_nu(:, :, 1)*theta0((np_lam+1):end));

thetapred = theta;
Wpred = W;

% Forward-Pass (Filtering)
for i=2:n_spk
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    lam(:, i) = exp(X_lam(:, :, i) * thetapred(1:np_lam, i));
    nu(:, i) = exp(G_nu(:, :, i)*thetapred((np_lam+1):end, i));
    
    INFO = zeros(size(W0));
    SCORE = zeros(length(theta0), 1);
    
    for j=1:size(lam, 1)
        
        logcum_app = logsum_calc(lam(j, i), nu(j, i), maxSum);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        log_C = logcum_app(4);
        log_D = logcum_app(5);
        log_E = logcum_app(6);
        
        mean_Y = exp(log_A - log_Z);
        var_Y = exp(log_B - log_Z) - mean_Y^2;
        mean_logYfac = exp(log_C - log_Z);
        var_logYfac = exp(log_D - log_Z) - mean_logYfac^2;
        cov_Y_logYfac =  exp(log_E-log_Z)-exp(log_A+log_C-2*log_Z);
        
%         cum_app = sum_calc(lam(j, i), nu(j, i), maxSum);
%         Z = cum_app(1);
%         A = cum_app(2);
%         B = cum_app(3);
%         C = cum_app(4);
%         D = cum_app(5);
%         E = cum_app(6);
%         
%         mean_Y = exp(log(A) - log(Z));
%         var_Y = exp(log(B) - log(Z)) - exp(2*log(mean_Y));
%         mean_logYfac = exp(log(C) - log(Z));
%         var_logYfac = exp(log(D) - log(Z)) - exp(2*log(mean_logYfac));
%         cov_Y_logYfac =  exp(log(E)-log(Z))-exp(log(A)+log(C)-2*log(Z));
        
        info1 = var_Y*X_lam(j,:,i)'*X_lam(j,:,i);
        info2 = -nu(j, i)*cov_Y_logYfac*X_lam(j,:,i)'*G_nu(j,:,i);
        info3 = info2';
        info4 = nu(j, i)*(nu(j, i)*var_logYfac - mean_logYfac + gammaln(N(j, i) + 1))*G_nu(j,:,i)'*G_nu(j,:,i);
        
        INFO = INFO + [info1, info2; info3, info4];
        SCORE = SCORE + [(N(j, i) - mean_Y)*X_lam(j,:,i)';...
            nu(j, i)*(-gammaln(N(j, i) + 1) + mean_logYfac)*G_nu(j,:,i)'];
        
    end
    
    Wpostinv = inv(Wpred(:,:,i)) + INFO;
    W(:,:,i) = inv(Wpostinv);
    
    theta(:,i)  = thetapred(:,i) + W(:,:,i)*SCORE;
    
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix')
        %return;
        break
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
end


end
