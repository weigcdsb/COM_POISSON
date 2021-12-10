function [mean_Y, var_Y, mean_logYfac, var_logYfac, cov_Y_logYfac, log_Z] = ...
    CMPmoment(lam, nu, maxSum)

if lam >= 2 && nu <= 2
    
    alpha = lam^(1/nu);
    c1 = (nu^2-1)/24;
    c2 = (nu^2-1)/48 + c1^2/2;
    
    log_Z = nu*alpha - (nu-1)*log(lam)/(2*nu) - (nu-1)*log(2*pi)/2 -...
        log(nu)/2 + log(1 + c1/(nu*alpha) + c2/(nu*alpha)^2);
    
    mean_Y = alpha - (nu-1)/(2*nu) - (nu^2-1)/(24*(nu^2)*alpha) - (nu^2-1)/(24*(nu^3)*(alpha^2));
    var_Y = alpha/nu + (nu^2-1)/(24*(nu^3)*alpha) + (nu^2-1)/(12*(nu^4)*(alpha^2));
    mean_logYfac = alpha*(log(lam)/nu - 1) + log(lam)/(2*nu^2) + 1/(2*nu) + log(2*pi)/2 -...
        (1/(24*alpha))*(1 + 1/nu^2 + log(lam)/nu - log(lam)/nu^3) -...
        (1/(24*alpha^2))*(1/nu^3 + log(lam)/nu^2 - log(lam)/nu^4);
    var_logYfac = alpha*(log(lam))^2/(nu^3) + log(lam)/nu^3 + 1/(2*nu^3) + ...
        (1/(24*nu^5*alpha))*(-2*nu^2 + 4*nu*log(lam) + (nu^2-1)*(log(lam))^2) +...
        (1/(24*nu^6*alpha^2))*(-3*nu^2 - 2*nu*(nu^2-3)*log(lam) + 2*(nu^2-1)*(log(lam))^2);
    cov_Y_logYfac = alpha*log(lam)/nu^2 + 1/(2*nu^2) +...
        (1/(24*alpha))*(2/nu^3 + log(lam)/nu^2 - log(lam)/nu^4) -...
        (1/(24*alpha^2))*(1/nu^2 - 3/nu^4 - 2*log(lam)/nu^3 + 2*log(lam)/nu^5);
    
else
    logcum_app = logsum_calc(lam, nu, maxSum);
    
    log_Z = logcum_app(1);
    mean_Y = exp(logcum_app(2) - log_Z);
    var_Y = exp(logcum_app(3) - log_Z) - mean_Y^2;
    mean_logYfac = exp(logcum_app(4) - log_Z);
    var_logYfac = exp(logcum_app(5) - log_Z) - mean_logYfac^2;
    cov_Y_logYfac =  exp(logcum_app(6)-log_Z)-exp(logcum_app(2)+logcum_app(4)-2*log_Z);
    
end


end