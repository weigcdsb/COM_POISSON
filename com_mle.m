function[theta] = com_mle(y, theta_lam0, theta_nu0, X_lam, G_nu, dt)

maxIter = 10000;
tol = 1e-12;

theta_lam_pre = theta_lam0;
theta_nu_pre = theta_nu0;
e = Inf;
lam
for k = 1:maxIter
    
    lam_pre = exp(X_lam(1, :).*theta_lam_pre)*dt;
    nu_pre = exp(G_nu(1, :).*theta_nu_pre)*dt;
    
    [grad, hess] = com_gradHess(y, lam_pre, nu_pre);
    [lam; nu] = [log(lam_pre); nu_pre] - inv(hess)*grad;
    e = sum((thetapre - theta).^2);
    thetapre = theta;
    if e < tol
        break
    end
end




end