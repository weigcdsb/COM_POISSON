function neg_llhd_pred = helper_window_v2_noNu(Q, theta0, N, X_lam, G_nu, W0, F, windSize, windType)

if(size(X_lam) >= 2)
    Q_lam = [Q(1) Q(2)*ones(1, size(X_lam, 2)-1)];
else
    Q_lam = Q(1);
end

Qmatrix = diag([Q_lam 0]);

[~,~, lam, nu, log_Zvec] =...
    ppafilt_compoisson_v2_window_fisher(theta0, N, X_lam, G_nu, W0, F, Qmatrix, windSize, windType);

if(length(log_Zvec) == size(N, 2))
    llhd_pred = sum(N.*log((lam+(lam==0))) -...
        nu.*gammaln(N + 1) - log_Zvec);
    fprintf('llhd %.02f... \n', llhd_pred);
    neg_llhd_pred = -llhd_pred;
else
    neg_llhd_pred = Inf;
end


end
