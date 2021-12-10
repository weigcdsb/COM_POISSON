function [lam, nu, logZ, CMP_mean, CMP_var] =...
    CMP_seq_calc(theta, X_lam, G_nu, nknot_lam, sumMax)

% for debug
% theta = theta_ho1;
% X_lam = Xb_ho;
% G_nu = Gb_ho_full;
% nknot_lam = nknots;
% sumMax = 1000;

T = size(theta,2);

lam = zeros(1,T);
nu = zeros(1,T);
logZ = zeros(1,T);
CMP_mean = zeros(1,T);
CMP_var = zeros(1,T);

for t = 1:T
    lam(t) = exp(X_lam(t,:)*theta(1:(nknot_lam+1), t));
    nu(t) = exp(G_nu(t,:)*theta((nknot_lam+2):end, t));
    [CMP_mean(t), CMP_var(t), ~, ~, ~, logZ(t)] = ...
            CMPmoment(lam(t), nu(t), 1000);
end

end