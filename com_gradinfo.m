function [grad, info] = com_gradinfo(y, theta, x_lam, g_nu, dt)

nCell = length(y);
np_lam = length(x_lam);
lam = exp(x_lam*theta(1:np_lam));
nu = exp(g_nu*theta((np_lam+1):end));

cum_app = sum_calc(lam*dt, nu*dt, 2*nCell);
Z = cum_app(1);
A = cum_app(2);
B = cum_app(3);
C = cum_app(4);
D = cum_app(5);
E = cum_app(6);

mean_Y = A/Z;
var_Y = B/Z - mean_Y^2;
mean_logYfac = C/Z;
var_logYfac = D/Z - mean_logYfac^2;
cov_Y_logYfac = E/Z - A*C/(Z^2);
sum_logfac = sum(gammaln(y + 1));

info1 = nCell*var_Y*(x_lam'*x_lam);
info2 = -nCell*cov_Y_logYfac*nu*dt*(x_lam'*g_nu);
info3 = -nCell*cov_Y_logYfac*nu*dt*(g_nu'*x_lam);
info4 = (nu*dt)*(nCell*nu*dt*var_logYfac)*...
    (g_nu'*g_nu);

info = [info1, info2; info3, info4];
grad = [(sum(y)- nCell*mean_Y)*x_lam';...
        nu*dt*(-sum_logfac + nCell*mean_logYfac)*g_nu'];

return