function [theta,W] = ppafilt_compoisson(N,X_lam,G_nu,theta0,W0,F,Q,dt)


n_spk = size(N, 2);
nCell = size(N, 1);
maxSum = n_spk; % max number for sum estimation;

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
    C = cum_app(3);
    D = cum_app(3);
    E = cum_app(3);
    
    lam1 = A/Z;
    lam2 = B/Z - lam1^2;
    nu1 = -nu(i)*dt*C/Z;
    nu2 = ((nu(i)*dt)^2*D)/Z + nu1 - nu1^2;
    lamnu = nu(i)*dt*(C*A/(Z^2) - E/Z);
    
    sum_logfac = sum(gammaln(N(:, i) + 1));
    
    w1 = nCell*lam2*(X_lam(i,:)'*X_lam(i,:));
    w2 = lamnu*(X_lam(i,:)'*G_nu(i, :));
    w3 = lamnu*(G_nu(i, :)'*X_lam(i,:));
    w4 = (nu(i)*dt*sum_logfac + nCell*nu2)*(G_nu(i, :)'*G_nu(i, :));
    
    
    
    Wpostinv = inv(Wpred(:,:,i)) + [w1, w2; w3, w4];
    W(:,:,i) = inv(Wpostinv);
%     W(:,:,i)
    
    theta(:,i)  = thetapred(:,i) +...
        W(:,:,i)*[(sum(N(:, i))- nCell*lam1)*X_lam(i,:)';...
        -(nu(i)*dt*sum_logfac + nCell*nu1)*G_nu(i, :)'];
    
%     [thetapred(:,i), theta(:,i),theta_true(i,:)']
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix')
        %return;
        break
    end
end

lastwarn('')

end