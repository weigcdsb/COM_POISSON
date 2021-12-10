

function [b,W,lam,lam_smoo] = ppasmoo_poissexp_nan(n,X,b0,W0,F,Q)

offset = n*0;

% Preallocate
b   = zeros(length(b0),length(n));
W   = zeros([size(W0) length(n)]);
lam = n*0;

% Initialize
b(:,1)   = b0;
W(:,:,1) = W0;
lam(1)   = exp(X(1,:)*b0 + offset(1));

bpred = b;
Wpred = W;

I = eye(size(X,2));

lastwarn('')
% Forward-Pass (Filtering)
for i=2:length(n)
    
    bpred(:,i) = F*b(:,i-1);
    Wpred(:,:,i) = F*W(:,:,i-1)*F' + Q;
    lam(i) = exp(X(i,:)*bpred(:,i) + offset(i));
    
    if ~isnan(n(i))
        Wpostinv = inv(Wpred(:,:,i)) + X(i,:)'*(lam(i))*X(i,:);
        W(:,:,i) = inv(Wpostinv);
        
        b(:,i)  = bpred(:,i) + W(:,:,i)*X(i,:)'*(n(i)-lam(i));
    else
        W(:,:,i) = Wpred(:,:,i);
        b(:,i)  = bpred(:,i);
    end
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        return;
    end
end

lam_smoo = lam;

% Backward-Pass (RTS)
for i=(length(n)-1):-1:1
    
    Wi = inv(Wpred(:,:,i+1));
    Fsquig = inv(F)*(I-Q*Wi);
    Ksquig = inv(F)*Q*Wi;
    
    theta(:,i)=Fsquig*b(:,i+1) + Ksquig*bpred(:,i+1);
    C = W(:,:,i)*F'*Wi;
    W(:,:,i) = W(:,:,i) + C*(W(:,:,i+1)-Wpred(:,:,i+1))*C';
    lam_smoo(i) = exp(X(i,:)*b(:,i) + offset(i));
end



end

