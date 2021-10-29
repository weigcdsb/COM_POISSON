function llhd = llhdCMP_consNu(nu, X_lam, theta, spk_vec)

maxSum = 10*max(spk_vec(:));
llhd = 0;
for t = 1:size(spk_vec, 2)
    loglamTmp = X_lam(t,:)*theta(:,t);
    lamTmp = exp(loglamTmp);
    [~, ~, ~, ~, ~, logZtmp] = CMPmoment(lamTmp, nu, maxSum);
    llhd = llhd + spk_vec(t)*loglamTmp - nu*gammaln(spk_vec(t)+1) - logZtmp;
end
end