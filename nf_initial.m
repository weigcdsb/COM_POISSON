function [theta] = nf_initial(y, x_lam, g_nu, dt)

maxIter = 100000;
tol = 1e-12;

% start by well-known approximation:
% (review paper: https://onlinelibrary.wiley.com/doi/10.1002/wics.1533)
% E(Y) = lam^(1/nu) - (nu-1)/2*nu
% Var(Y) = lam^(1/nu)/nu

% ignoring (nu-1)/2*nu in E(Y) yields
% nu = E(Y)/Var(Y) = mean(Y)/var(Y)

% delete the outlier (Q1 - 1.5IQR <= y <= Q3 + 1.5IQR)
y = rmoutliers(y, 'quartiles');
theta0 = [((x_lam*x_lam')^-1)*x_lam*(log(mean(y)) - log(dt));...
    ((g_nu*g_nu')^-1)*g_nu*(log(mean(y)) -log(var(y)) - log(dt))];

thetaPre = theta0;
for k = 1:maxIter
    [grad, hess] = com_gradHess(y, thetaPre, x_lam, g_nu, dt);
    theta = thetaPre - inv(hess)*grad;
    e = sum((theta - thetaPre).^2);
    thetaPre = theta;
    if e < tol
        break
    end
end


end