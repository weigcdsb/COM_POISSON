function [theta,W] = ppasmoo_compoisson(N,X_lam,G_nu,W0,F,Q,dt)

% newton raphson
% theta0 = nf_initial(N(:, 1), X_lam(1,:), G_nu(1,:), dt);
theta0 = nf_fisher_initial(N(:, 1), X_lam(1,:), G_nu(1,:), dt);

n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = 5*n_spk; % max number for sum estimation;

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

% Forward-Pass (Filtering)
for i=2:n_spk
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    
    lam(i) = exp(X_lam(i,:)*thetapred(1:np_lam, i));
    nu(i) = exp(G_nu(i,:)*thetapred((np_lam+1):end, i));
    
    cum_app = sum_calc(lam(i)*dt, nu(i)*dt, maxSum);
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
    sum_logfac = sum(gammaln(N(:, i) + 1));
    
    info1 = nCell*var_Y*(X_lam(i,:)'*X_lam(i,:));
    info2 = -nCell*cov_Y_logYfac*nu(i)*dt*(X_lam(i,:)'*G_nu(i, :));
    info3 = -nCell*cov_Y_logYfac*nu(i)*dt*(G_nu(i, :)'*X_lam(i,:));
    info4 = (nu(i)*dt)*(nCell*nu(i)*dt*var_logYfac-nCell*mean_logYfac + sum_logfac)*...
        (G_nu(i, :)'*G_nu(i, :));
    
    Wpostinv = inv(Wpred(:,:,i)) + [info1, info2; info3, info4];
    W(:,:,i) = inv(Wpostinv);
    
    theta(:,i)  = thetapred(:,i) +...
        W(:,:,i)*[(sum(N(:, i))- nCell*mean_Y)*X_lam(i,:)';...
        nu(i)*dt*(-sum_logfac + nCell*mean_logYfac)*G_nu(i, :)'];
    
%     [thetapred(:,i), theta(:,i),theta_true(i,:)']
    
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
