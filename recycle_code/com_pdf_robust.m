
function pdf = com_pdf_robust(y, lambda, nu, summax)

n=0:summax;
log_p = (n*log(lambda) - (nu*gammaln(n+1)));
log_pmax = max(log_p);
lse = log_pmax+log(sum(exp(log_p-log_pmax)));

log_py = (y*log(lambda) - (nu*gammaln(y+1)));
pdf = exp(log_py-lse);