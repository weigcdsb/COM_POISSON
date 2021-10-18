function [mean_grid, var_grid, ff_grid] = cmp_grid(theta, nknots, X_lam, G_nu, sumMax)

lam_grid = exp(X_lam*theta(1:(nknots+1),:));
nu_grid = exp(G_nu*theta((nknots+2):end,:));
mean_grid = zeros(size(lam_grid));
var_grid = zeros(size(lam_grid));
for m = 1:size(mean_grid,1)
    for n = 1:size(mean_grid,2)
        [mean_grid(m,n), var_grid(m,n), ~, ~, ~, ~] = ...
            CMPmoment(lam_grid(m,n), nu_grid(m,n), sumMax);
    end
end

ff_grid = var_grid./mean_grid;

end


