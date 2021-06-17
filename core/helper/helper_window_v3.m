function neg_llhd_pred = helper_window_v3(Q, theta0, N, X_lam, G_nu, W0, F, windSize, windType)

if(size(X_lam, 2) >= 2)
    Q_lam = [Q(1) Q(2)*ones(1, size(X_lam, 2)-1)];
    if(size(G_nu, 2) >= 2)
        Q_nu = [Q(3) Q(4)*ones(1, size(G_nu, 2) - 1)];
    else
        Q_nu = Q(3);
    end
else
    Q_lam = Q(1);
    if(size(G_nu, 2) >= 2)
        Q_nu = [Q(2) Q(3)*ones(1, size(G_nu, 2) - 1)];
    else
        Q_nu = Q(2);
    end
end


Qmatrix = diag([Q_lam Q_nu]);
Qmatrix(1, length(Q_lam)+1) = Q(end);
Qmatrix(length(Q_lam)+1, 1) = Q(end);

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