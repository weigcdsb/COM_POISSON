function mse = CMPmse(spk_vec, lam, nu)

mean_Y = zeros(size(lam));
for t = 1:length(lam)
    [mean_Y(t), ~, ~, ~, ~, ~] = CMPmoment(lam(t), nu(t), 1000);
end
spk_vec = spk_vec(:)';
mean_Y = mean_Y(:)';

mse = mean((spk_vec - mean_Y).^2);

end
