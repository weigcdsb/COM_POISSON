function [theta,W,lam_smoo] = ppasmoo_cmp_fixNu_na(N,X_lam,nu,theta0,W0,F,Q)

% to debug
% N = spk_vec;
% X_lam = Xb;
% nu = nu_trace(g-1);
% theta0 = theta02(1:(end-1));
% W0 = W02(1:(end-1), 1:(end-1));
% F = eye(size(Xb, 2));
% Q = Qoptmatrix3;

n_spk = size(N, 2);
maxSum = 10*max(N(:)); % max number for sum estimation;

% Preallocate
theta   = zeros(length(theta0),length(N));
W   = zeros([size(W0) length(N)]);
lam = N*0;

% Initialize
theta(:,1)   = theta0;
W(:,:,1) = W0;
lam(1)   = exp(X_lam(1,:)*theta0);

thetapred = theta;
Wpred = W;

I = eye(length(theta0));
lastwarn('')

% Forward-pass: filtering
for i = 2:n_spk
    
    thetapred(:,i) = F*theta(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    lam(i) = exp(X_lam(i,:)*thetapred(:,i));
    
    if ~isnan(N(:, i))
        [mean_Y, var_Y, ~, ~, ~, ~] = CMPmoment(lam(i), nu, maxSum);
        
        Wpostinv = inv(Wpred(:,:,i)) + var_Y*X_lam(i,:)'*X_lam(i,:);
        W(:,:,i) = inv(Wpostinv);
        theta(:,i)  = thetapred(:,i) + W(:,:,i)*(N(:, i) - mean_Y)*X_lam(i,:)';
    else
        W(:,:,i) = Wpred(:,:,i);
        theta(:,i)  = thetapred(:,i);
    end
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        return;
    end
    
end

lam_smoo = lam;

% Backward-Pass (RTS)
for i=(n_spk-1):-1:1
    
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(F)*(I-Q*Wi);
    Ksquig = inv(F)*Q*Wi;
    
    theta(:,i)=Fsquig*theta(:,i+1) + Ksquig*thetapred(:,i+1);
    C = W(:,:,i)*F'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    
    
    lam_smoo(i) = exp(X_lam(i,:)*theta(:, i));
end


end