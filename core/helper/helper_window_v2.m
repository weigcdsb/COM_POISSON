function neg_llhd_pred = helper_window_v2(p, N, X_lam, G_nu, F, theta0Set, W0Set, nWindLB)

Qmatrix = diag(p(1:(end-1)));
windSize = p(end);
fprintf('windowSize... %d...', windSize)

theta0 = theta0Set(:, (windSize -nWindLB + 1));
W0 = W0Set(:, :, (windSize -nWindLB + 1));

[~,~, lam, nu, log_Zvec] =...
    ppafilt_compoisson_v2_window(theta0, N, X_lam, G_nu, W0, F, Qmatrix, windSize);

if(length(log_Zvec) == size(N, 2))
    llhd_pred = sum(N.*log((lam+(lam==0))) -...
        nu.*gammaln(N + 1) - log_Zvec);
    fprintf('llhd %.02f... \n', llhd_pred);
    neg_llhd_pred = -llhd_pred;
else
    neg_llhd_pred = Inf;
end