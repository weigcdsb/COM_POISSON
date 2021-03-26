function[llhd, g] = com_llhd2(y, lambda, nu, negllhd)

summax = 1000;
termlim = 1e-12;
cum_app = zeros(1, 3);

for js = 1:summax
    
    term = [exp((js-1)*log(lambda) - nu*gammaln(js)),...
        exp(log(js-1) + (js-1)*log(lambda) - nu*gammaln(js)),...
        exp((js-1)*log(lambda) - nu*gammaln(js))*gammaln(js)];
    
    if(js > 5)
        if((max(term)/min(cum_app)) < termlim)
            break
        end
    end
    cum_app = cum_app + term;
end

Z = cum_app(1);
A = cum_app(2);
C = cum_app(3);

llhd = log(lambda)*sum(y) - nu*sum(gammaln(y + 1)) - length(y)*log(Z);
if(negllhd)
    llhd = -llhd;
end

if(nargout > 1)
    mean_Y = A/Z;
    mean_logYfac = C/Z;
    g = zeros(2, 1);
    g(1) = (1/lambda)*(sum(y) - length(y)*mean_Y);
    g(2) = length(y)*mean_logYfac - sum(gammaln(y + 1));
    if(negllhd)
        g= -g;
    end
end


return