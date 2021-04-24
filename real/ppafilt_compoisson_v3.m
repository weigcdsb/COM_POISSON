function [theta,W] = ppafilt_compoisson_v3(theta0, N,X_lam,G_nu,W0,F,Q)

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 5*n_spk; % max number for sum estimation;

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
    nu(:, i) = exp(G_nu(:, :, i)*theta0((np_lam+1):end));
    
    INFO = zeros(size(W0));
    SCORE = zeros(length(theta0), 1);
    for j=1:size(lam, 1)
        cum_app = sum_calc(lam(j, i), nu(j, i), maxSum);
        Z = cum_app(1);
        A = cum_app(2);
        B = cum_app(3);
        C = cum_app(4);
        D = cum_app(5);
        E = cum_app(6);
        
        mean_Y = A/Z;
        var_Y = B/Z - mean_Y^2;
        mean_logYfac = C/Z;
        var_logYfac = D/Z - mean_logYfac^2;
        cov_Y_logYfac = E/Z - A*C/(Z^2);
        
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

end
