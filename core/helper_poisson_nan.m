function neg_llhd_pred = helper_poisson_nan(Q, b0, N, X, W0, F, varargin)

if(size(X, 2) >= 2)
    Qmatrix = diag([Q(1) Q(2)*ones(1, size(X, 2)-1)]);
else
    Qmatrix = Q;
end

[~,~,lam] = ppafilt_poissexp_nan(N,X,b0,W0,F,Qmatrix);
llhd_pred = nansum(-lam + log((lam+(lam==0))).*N - gammaln(N + 1));
fprintf('llhd %.02f... \n', llhd_pred);
neg_llhd_pred = -llhd_pred;

end