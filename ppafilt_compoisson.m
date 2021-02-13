function [theta,W,lam, nu, i] = ppafilt_compoisson(N,X_lam,G_nu,theta0,W0,F,Q,dt)


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
    Zk = cum_app(1);
    Ak = cum_app(2);
    Bk = cum_app(3);
    
    mean_k = Ak/Zk;
    var_k = Bk/Zk - mean_k^2;
    sum_logfac = sum(gammaln(N(:, i) + 1));
    
    w_lam = nCell*var_k*(X_lam(i,:)'*X_lam(i,:));
    w_nu = nu(i)*dt*sum_logfac*(G_nu(i, :)'*G_nu(i, :));
    
    Wpostinv = inv(Wpred(:,:,i)) + diag([w_lam, w_nu]);
    W(:,:,i) = inv(Wpostinv);
%     W(:,:,i)
    
    theta(:,i)  = thetapred(:,i) +...
        W(:,:,i)*[(sum(N(:, i))- nCell*mean_k)*X_lam(i,:)';...
        -((nu(i)*dt*sum_logfac))*G_nu(i, :)'];
    
%     [theta(:,i),theta_true(i,:)']
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:nearlySingularMatrix')
        %return;
        break
    end
end

lastwarn('')

end
