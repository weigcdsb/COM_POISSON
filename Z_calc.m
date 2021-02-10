function sum1 = Z_calc(lam, nu, n)

termlim = 1e-6;
sum1 = 0;
for js = 1:n
    term = exp(log(lam^(js-1)) - nu*gammaln(js));
    if(js > 3)
        if((term/sum1) < termlim)
            break
        end
    end
    sum1 = sum1 + term;
end

end