function [theta,W,lam_smoo] = ppasmoo_cmp_fixNu(N,X_lam,nu,theta0,W0,F,Q, varargin)

% to debug
% N = spk_vec;
% nu = nu_trace(1);
% W0 = 1e-2;
% F = 1;
% Q = 1e-3;
% maxSum = 1000;
n_spk = size(N, 2);
obsIdxAll = 1:n_spk;
maxSum = 10*max(N(:)); % max number for sum estimation;

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'obsIdx'}
                obsIdxAll = varargin{c+1};
            case {'offset'}
                offset = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

h = diff(obsIdxAll);

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
for i = 2:length(N)
    Ftmp = F^(h(i-1));
    Qtmp = 0;
    for l = 1:h(i-1)
        Qtmp = Qtmp + (F^(l-1))*Q*(F^(l-1))';
    end
    
    thetapred(:,i) = Ftmp*theta(:,i-1);
    Wpred(:,:,i) = Ftmp*W(:,:,i-1)*Ftmp' + Qtmp;
    
    lam(i) = exp(X_lam(i,:)*thetapred(:,i));
    
    [mean_Y, var_Y, ~, ~, ~, ~] = ...
            CMPmoment(lam(i), nu, maxSum);
    
    Wpostinv = inv(Wpred(:,:,i)) + var_Y*X_lam(i,:)'*X_lam(i,:);
    W(:,:,i) = inv(Wpostinv);
    theta(:,i)  = thetapred(:,i) + W(:,:,i)*(N(:, i) - mean_Y)*X_lam(i,:)';
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        return;
    end
    
end

lam_smoo = lam;

% Backward-Pass (RTS)
for i=(n_spk-1):-1:1
    Ftmp = F^(h(i));
    Qtmp = 0;
    for l = 1:h(i)
        Qtmp = Qtmp + (F^(l-1))*Q*(F^(l-1))';
    end
    
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(Ftmp)*(I-Qtmp*Wi);
    Ksquig = inv(Ftmp)*Qtmp*Wi;
    
    theta(:,i)=Fsquig*theta(:,i+1) + Ksquig*thetapred(:,i+1);
    C = W(:,:,i)*Ftmp'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    
    lam_smoo(i) = exp(X_lam(i,:)*theta(:, i));
end


end