function cum_log_app = logsum_calc(lam, nu, n)

cum_log_app = zeros(1, 6);

js=0:n;
log_Zi = js*log(lam) - nu*gammaln(js+1);
log_Ai = log(js) + (js)*log(lam) - nu*gammaln(js+1);
log_Bi = 2*log(js) + (js)*log(lam) - nu*gammaln(js+1);
log_Ci = log_Zi(3:(n+1)) + log(gammaln((2:n) + 1));
log_Di = log_Zi(3:(n+1)) + 2*log(gammaln((2:n) + 1));
log_Ei = log_Ai(3:(n+1)) + log(gammaln((2:n) + 1));

log_Zmax = max(log_Zi);
log_Amax = max(log_Ai);
log_Bmax = max(log_Bi);
log_Cmax = max(log_Ci);
log_Dmax = max(log_Di);
log_Emax = max(log_Ei);

cum_log_app(1) = log_Zmax+log(sum(exp(log_Zi-log_Zmax)));
cum_log_app(2) = log_Amax+log(sum(exp(log_Ai-log_Amax)));
cum_log_app(3) = log_Bmax+log(sum(exp(log_Bi-log_Bmax)));
cum_log_app(4) = log_Cmax+log(sum(exp(log_Ci-log_Cmax)));
cum_log_app(5) = log_Dmax+log(sum(exp(log_Di-log_Dmax)));
cum_log_app(6) = log_Emax+log(sum(exp(log_Ei-log_Emax)));


end