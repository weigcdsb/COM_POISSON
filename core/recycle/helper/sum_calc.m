function cum_app = sum_calc(lam, nu, n)

termlim = 1e-12;
cum_app = zeros(1, 6);

for js = 1:n
    
    term = [exp((js-1)*log(lam) - nu*gammaln(js)),...
         exp(log(js-1) + (js-1)*log(lam) - nu*gammaln(js)),...
       exp( 2*log(js-1) + (js-1)*log(lam) - nu*gammaln(js)),...
       exp((js-1)*log(lam) - nu*gammaln(js))*gammaln(js),...
       exp((js-1)*log(lam) - nu*gammaln(js))*(gammaln(js))^2,...
       exp(log(js-1) + (js-1)*log(lam) - nu*gammaln(js))*gammaln(js)];
    
    if(js > 5 && ~isnan(max(term)/min(cum_app)))
        if((max(term)/min(cum_app)) < termlim)
            break
        end
    end
    cum_app = cum_app + term;
end

end