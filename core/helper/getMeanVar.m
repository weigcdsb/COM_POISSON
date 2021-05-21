
function [theo_mean,theo_var]=getMeanVar(lam,nu)

theo_mean=zeros(size(lam));
theo_var=zeros(size(lam));
for k=1:length(lam)
    logcum_app = logsum_calc(lam(k), nu(k), 1000);
    log_Zk = logcum_app(1);
    log_Ak = logcum_app(2);
    log_Bk = logcum_app(3);

    theo_mean(k) = exp(log_Ak - log_Zk);
    theo_var(k) = exp(log_Bk - log_Zk) - theo_mean(k)^2;
end