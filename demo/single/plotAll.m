function plotAll(idx, spk_vec, X_lam, G_nu, theta_true, theta_fit)

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);
[est_mean,est_var]=getMeanVar(exp(theta_fit(1,:)),exp(theta_fit(2,:)));

figure(idx)
subplot(2,3,1)
plot(mean(spk_vec, 1))
hold on
plot(theo_mean, 'r', 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')

subplot(2,3,2)
plot(theo_mean)
hold on
plot(est_mean)
hold off
ylabel('Mean')

subplot(2,3,5)
plot(theo_var)
hold on
plot(est_var)
hold off
ylabel('Var')


subplot(2,3,4)
plot(theo_var./theo_mean)
hold on
plot(est_var./est_mean)
hold off
ylim([0 4])
ylabel('Fano Factor')

subplot(2,3,3)
plot(theta_true(:,1))
hold on
plot(theta_fit(1,:))
hold off
ylabel('beta')

subplot(2,3,6)
plot(theta_true(:,2))
hold on
plot(theta_fit(2,:))
hold off
ylabel('gamma')

end