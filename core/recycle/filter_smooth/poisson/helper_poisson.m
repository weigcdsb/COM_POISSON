function neg_llhd_pred = helper_poisson(Q, b0, N, X, W0, F, varargin)

obsIdxAll = 1:length(N);

if (~isempty(varargin))
    c = 1 ;
    while c <= length(varargin)
        switch varargin{c}
            case {'obsIdx'}
                obsIdxAll = varargin{c+1};
        end % switch
        c = c + 2;
    end % for
end % if

if(size(X, 2) >= 2)
    Qmatrix = diag([Q(1) Q(2)*ones(1, size(X, 2)-1)]);
else
    Qmatrix = Q;
end


[~,~,lam] = ppafilt_poissexp(N,X,b0,W0,F,Qmatrix,'obsIdx', obsIdxAll);
llhd_pred = sum(-lam + log((lam+(lam==0))).*N - gammaln(N + 1));
fprintf('llhd %.02f... \n', llhd_pred);
neg_llhd_pred = -llhd_pred;

end