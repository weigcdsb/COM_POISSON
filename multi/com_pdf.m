% com_pdf.m - compute Conway-Maxwell-Poisson Probability Density Function.
%   See "Conjugate Analysis of the Conway-Maxwell-Poisson Distribution", 
%   J. Kadane et al., Carnegie Mellon et al., 6/27/20031.
%
%   Created by:     J. Huntley,  07/15/05
%

function[pdf] = com_pdf(n, lambda, nu)

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

pdf = exp(n*log(lambda) - (nu*gammaln(n+1)))/Z;

return