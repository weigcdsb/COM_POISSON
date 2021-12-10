function plotAll_filtSmoo(X_lam, G_nu, theta_true, theta_fit1, theta_fit2, W_fit1, W_fit2)

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);
[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));
[est_mean2,est_var2]=getMeanVar(exp(theta_fit2(1,:)),exp(theta_fit2(2,:)));
[var_rate_exact1, ~] = varCE(X_lam, G_nu, theta_fit1, W_fit1);
[var_rate_exact2, ~] = varCE(X_lam, G_nu, theta_fit2, W_fit2);


tiledlayout(2,3, 'TileSpacing', 'compact')
nexttile
theo_ff = theo_var./theo_mean;
plot(theo_ff, 'k');
hold on
plot(est_var1./est_mean1, 'r')
plot(est_var2./est_mean2, 'b')
hold off
ylim([0 4])
ylabel('Fano Factor')
ylim([0 round(max(theo_ff)) + 1.5 ])


nexttile
line1 = plot(theo_mean, 'k');
hold on
line2 = plot(est_mean1, 'r');
line3 =plot(est_mean2, 'b');
hold off
ylabel('Mean')
ylim([round(min(theo_mean)) - 5 round(max(theo_mean)) + 5 ])


nexttile
plot((theta_true(:,1)), 'k');
hold on
plot((theta_fit1(1,:)), 'r')
plot((theta_fit2(1,:)), 'b')
hold off
ylabel('beta')
ylim([round(min(theta_true(:,1))) - 5 round(max(theta_true(:,1))) + 5 ])


nexttile
hold on
plot(var_rate_exact1, 'r')
plot(var_rate_exact2, 'b')
ylabel('varCE')
ylim([0 max([var_rate_exact1(100:(end - 100)); var_rate_exact2(100:(end - 100))]) + .5])
hold off


nexttile
plot(theo_var, 'k');
hold on
plot(est_var1, 'r')
plot(est_var2, 'b')
hold off
ylabel('Var')
ylim([round(min(theo_var)) - 5 round(max(theo_var)) + 5 ])


nexttile
plot((theta_true(:,2)), 'k');
hold on
plot((theta_fit1(2,:)), 'r')
plot((theta_fit2(2,:)), 'b')
hold off
ylabel('gamma')
ylim([round(min(theta_true(:,2))) - 5 round(max(theta_true(:,2))) + 5 ])


lg = legend(nexttile(3), [line1,line2,line3], {'true', 'filtering', 'smoothing'});
lg.Location = 'northeastoutside';

end
