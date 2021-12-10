function plotAll_filtSmoo_v2(theta_true, theta_fit1, theta_fit2)

tiledlayout(1,2, 'TileSpacing', 'compact')
nexttile
line1 = plot((theta_true(:,1)), 'k');
hold on
line2 = plot((theta_fit1(1,:)), 'r');
line3 = plot((theta_fit2(1,:)), 'b');
hold off
ylabel('beta')
ylim([round(min(theta_true(:,1))) - 5 round(max(theta_true(:,1))) + 5 ])


nexttile
plot((theta_true(:,2)), 'k');
hold on
plot((theta_fit1(2,:)), 'r')
plot((theta_fit2(2,:)), 'b')
hold off
ylabel('gamma')
ylim([round(min(theta_true(:,2))) - 5 round(max(theta_true(:,2))) + 5 ])


lg = legend(nexttile(2), [line1,line2,line3], {'true', 'filtering', 'smoothing'});
lg.Location = 'northeastoutside';

end
