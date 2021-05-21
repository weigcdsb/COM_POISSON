function neg_llhd_pred = helper_2d(Q, theta0, N, X_lam, G_nu, W0, F)

Qmatrix = diag(Q);

[~,~, lam, nu, log_Zvec] = ppafilt_compoisson_v2(theta0, N, X_lam, G_nu, W0, F, Qmatrix);
llhd_pred = sum(N.*log((lam+(lam==0))) -...
            nu.*gammaln(N + 1) - log_Zvec);
fprintf('llhd %.02f... \n', llhd_pred);
neg_llhd_pred = -llhd_pred;

end


