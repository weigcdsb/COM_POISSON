% Point-process adaptive smoothing w/ Poisson likelihood (log-link)
%  filtering via Eden et al. Neural Comp 2004
%  then a backward pass based on Rauch-Tung-Striebel

function [b,W,lam] = ppafilt_poissexp_nan(n,X,b0,W0,F,Q)

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

end
