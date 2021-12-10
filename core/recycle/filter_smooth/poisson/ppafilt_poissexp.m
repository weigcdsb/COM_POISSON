% Point-process adaptive smoothing w/ Poisson likelihood (log-link)
%  filtering via Eden et al. Neural Comp 2004
%  then a backward pass based on Rauch-Tung-Striebel

function [b,W,lam] = ppafilt_poissexp(n,X,b0,W0,F,Q, varargin)

offset = n*0;
obsIdxAll = 1:length(n);

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
    
    Ftmp = F^(h(i-1));
    Qtmp = 0;
    for l = 1:h(i-1)
        Qtmp = Qtmp + (F^(l-1))*Q*(F^(l-1))';
    end
    
    bpred(:,i) = Ftmp*b(:,i-1);
    Wpred(:,:,i) = Ftmp*W(:,:,i-1)*Ftmp' + Qtmp;
    
    lam(i) = exp(X(i,:)*bpred(:,i) + offset(i));
    
    Wpostinv = inv(Wpred(:,:,i)) + X(i,:)'*(lam(i))*X(i,:);
    W(:,:,i) = inv(Wpostinv);
    
    b(:,i)  = bpred(:,i) + W(:,:,i)*X(i,:)'*(n(i)-lam(i));
    
    [~, msgid] = lastwarn;
    if strcmp(msgid,'MATLAB:illConditionedMatrix')
        return;
    end
end

end
