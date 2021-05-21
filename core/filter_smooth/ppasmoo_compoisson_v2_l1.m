function [theta,W] = ppasmoo_compoisson_v2_l1(theta0, N,X_lam,G_nu,W0,F,Q, zeta)

% newton raphson
% theta0 = nf_initial(N(:, 1), X_lam(1,:), G_nu(1,:), dt);
% theta0 = nf_fisher_initial(N(:, 1), X_lam(1,:), G_nu(1,:), dt);

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 10*max(N(:)); % max number for sum estimation;

% Preallocate
theta   = zeros(length(theta0), n_spk);
W   = zeros([size(W0) n_spk]);
lam = n_spk*0;
nu = n_spk*0;
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
    
    lam(i) = exp(X_lam(i,:)*thetapred(1:np_lam, i));
    nu(i) = exp(G_nu(i,:)*thetapred((np_lam+1):end, i));
    
    logcum_app = logsum_calc(lam(i), nu(i), maxSum);
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
    
    sum_logfac = sum(gammaln(N(:, i) + 1));
    
    info1 = nCell*var_Y*(X_lam(i,:)'*X_lam(i,:));
    info2 = -nCell*cov_Y_logYfac*nu(i)*(X_lam(i,:)'*G_nu(i, :));
    info3 = -nCell*cov_Y_logYfac*nu(i)*(G_nu(i, :)'*X_lam(i,:));
    info4 = nu(i)*(nCell*nu(i)*var_logYfac-nCell*mean_logYfac + sum_logfac)*...
        (G_nu(i, :)'*G_nu(i, :));
    
    Wpostinv = inv(Wpred(:,:,i)) + [info1, info2; info3, info4];
    W(:,:,i) = inv(Wpostinv);
    
    theta(:,i)  = thetapred(:,i) +...
        W(:,:,i)*([(sum(N(:, i))- nCell*mean_Y)*X_lam(i,:)';...
        nu(i)*(-sum_logfac + nCell*mean_logYfac)*G_nu(i, :)'] + ...
        zeta*sum(thetapred(:,i) ~= 0)); % penalized term
    
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
end


end
