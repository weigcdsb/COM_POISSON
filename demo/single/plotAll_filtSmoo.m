function plotAll_filtSmoo(spk_vec, X_lam, G_nu, theta_true, theta_fit1, theta_fit2)

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);
[est_mean1,est_var1]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));
[est_mean2,est_var2]=getMeanVar(exp(theta_fit2(1,:)),exp(theta_fit2(2,:)));


tiledlayout(2,3, 'TileSpacing', 'compact')
nexttile
plot(mean(spk_vec, 1));
hold on
plot(theo_mean, 'r', 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')

nexttile
line1 = plot(theo_mean, 'k');
hold on
line2 = plot(est_mean1, 'r');
line3 =plot(est_mean2, 'b');
hold off
ylabel('Mean')

nexttile
plot((theta_true(:,1)), 'k');
hold on
plot((theta_fit1(1,:)), 'r')
plot((theta_fit2(1,:)), 'b')
hold off
ylabel('beta')

nexttile
plot(theo_var./theo_mean, 'k');
hold on
plot(est_var1./est_mean1, 'r')
plot(est_var2./est_mean2, 'b')
hold off
ylim([0 4])
ylabel('Fano Factor')

nexttile
plot(theo_var, 'k');
hold on
plot(est_var1, 'r')
plot(est_var2, 'b')
hold off
ylabel('Var')

nexttile
plot((theta_true(:,2)), 'k');
hold on
plot((theta_fit1(2,:)), 'r')
plot((theta_fit2(2,:)), 'b')
hold off
ylabel('gamma')

lg = legend(nexttile(3), [line1,line2,line3], {'true', 'filtering', 'smoothing'});
lg.Location = 'northeastoutside';

end
