function [var_rate_exact, var_rate_app] = varParam(X_lam, G_nu, theta_fit, W_fit)

maxSum = 1000;
var_rate_exact = zeros(size(theta_fit, 2), 1);
var_rate_app = zeros(size(theta_fit, 2), 1);

nLam = size(X_lam, 2);
nNu = size(G_nu, 2);

for k = 1:size(X_lam, 1)
    x_tmp = X_lam(k, :);
    g_tmp = G_nu(k, :);
    Z_tmp = zeros(2, nLam + nNu);
    Z_tmp(1, 1:nLam) = x_tmp;
    Z_tmp(2, (nLam+1):end) = g_tmp;
    
    Mu_tmp = Z_tmp*theta_fit(:, k);
    Sig_tmp = Z_tmp*W_fit(:, :, k)*Z_tmp';
    
    V = (exp(Sig_tmp) - 1).*...
        exp([Mu_tmp ones(2, 1)]*[ones(1, 2); Mu_tmp'] +...
        1/2*([diag(Sig_tmp) ones(2, 1)]*[ones(1, 2); diag(Sig_tmp)']));
    
    lam_tmp = exp(Mu_tmp(1));
    nu_tmp = exp(Mu_tmp(2));
    
    % exact
    [~, var_Y, ~, ~, cov_Y_logYfac, ~] = ...
    CMPmoment(lam_tmp, nu_tmp, maxSum);
    
    grad_tmp_exact = [var_Y/lam_tmp -cov_Y_logYfac];
    var_rate_exact(k) = grad_tmp_exact * V *grad_tmp_exact';
    
    % approximation
    grad_tmp_app = [(1/nu_tmp)*(lam_tmp^(1/nu_tmp - 1)) ...
        -(lam_tmp^(1/nu_tmp))*log(lam_tmp)/(nu_tmp^2)-...
        1/(2*nu_tmp^2)];
    var_rate_app(k) = grad_tmp_app * V *grad_tmp_app';
    
end

end