function [grad, hess] = com_gradHess(y, theta, x_lam, g_nu, dt)

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

lam1 = A/Z;
lam2 = B/Z - lam1^2;
nu1 = -nu*dt*C/Z;
nu2 = ((nu*dt)^2*D)/Z + nu1 - nu1^2;
lamnu = nu*dt*(C*A/(Z^2) - E/Z);

sum_logfac = sum(gammaln(y + 1));

w1 = nCell*lam2*(x_lam'*x_lam);
w2 = lamnu*(x_lam'*g_nu);
w3 = lamnu*(g_nu'*x_lam);
w4 = (nu*dt*sum_logfac + nCell*nu2)*(g_nu'*g_nu);

hess = -[w1, w2; w3, w4];
grad = [(sum(y)- nCell*lam1)*x_lam';...
        -(nu*dt*sum_logfac + nCell*nu1)*g_nu'];

return