function llhd = CMPllhd(spk_vec, lam, nu)

logZ = zeros(size(lam));
for t = 1:length(lam)
    [~, ~, ~, ~, ~, logZ(t)] = CMPmoment(lam(t), nu(t), 1000);
end
spk_vec = spk_vec(:)';
lam = lam(:)';
nu = nu(:)';
llhd = sum(spk_vec.*log((lam+(lam==0))) -...
        nu.*gammaln(spk_vec + 1) - logZ);

end



