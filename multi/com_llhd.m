function[llhd, g] = com_llhd(y, lambda, nu)

summax = 1000; 
termlim = 1e-12;
Z = 0;
for js = 1:summax
    term = exp((js-1)*log(lambda) - nu*gammaln(js));
    if(js > 3)
        if((term/Z) < termlim)
            break
        end
    end
    Z = Z + term;
end

llhd = log(lambda)*sum(y) - nu*sum(gammaln(y + 1)) - length(y)*Z;

if


return