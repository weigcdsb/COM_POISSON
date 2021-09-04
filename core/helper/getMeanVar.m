
function [theo_mean,theo_var]=getMeanVar(lam,nu)

theo_mean=zeros(size(lam));
theo_var=zeros(size(lam));
for k=1:length(lam)
    [theo_mean(k), theo_var(k)] = CMPmoment(lam(k), nu(k), 1000);
end