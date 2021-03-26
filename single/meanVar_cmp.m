function [mu, sig2] = meanVar_cmp(lam, nu, maxSum)

cum_app = sum_calc(lam, nu, maxSum);
Z = cum_app(1);
A = cum_app(2);
B = cum_app(3);

mu = A/Z;
sig2 = B/Z - mu^2;

end