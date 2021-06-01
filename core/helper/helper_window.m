function neg_llhd_pred = helper_window(Q, theta0, N, X_lam, G_nu, W0, F, windSize)

Qmatrix = diag(Q);

[~,~, lam, nu, log_Zvec] =...
    ppafilt_compoisson_v2_window_fisher(theta0, N, X_lam, G_nu, W0, F, Qmatrix, windSize);

if(length(log_Zvec) == size(N, 2))
    llhd_pred = sum(N.*log((lam+(lam==0))) -...
        nu.*gammaln(N + 1) - log_Zvec);
    fprintf('llhd %.02f... \n', llhd_pred);
    neg_llhd_pred = -llhd_pred;
else
    neg_llhd_pred = Inf;
end


end


