function GradHess = gradHessNu(nu, X_lam, theta, spk_vec)

% to debug
% nu = nu_trace(1);
% theta = theta_lam(:,:,g);

maxSum = 10*max(spk_vec(:));
grad = 0;
hess = 0;

for t = 1:size(spk_vec, 2)
    lamTmp = exp(X_lam(t,:)*theta(:,t));
    [~, ~, mean_logYfac, var_logYfac, ~, ~] = ...
        CMPmoment(lamTmp, nu, maxSum);
    
    grad = grad + (-gammaln(spk_vec(t)+1) + mean_logYfac);
    hess = hess - var_logYfac;
end

GradHess{1} = grad;
GradHess{2} = hess;

end