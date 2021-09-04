addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%%
rng(2)
k = 200;
X_lam = ones(k, 1);
G_nu = ones(k, 1);

theta_true = zeros(2,k);
% betaPre = 0.5; betaPost = 3;
% gamPre = -1.5; gamPost = 2;
% theta_true(1,:) = [ones(1, round(k/2))*betaPre ones(1, k-round(k/2))*betaPost];
% theta_true(2,:) = [ones(1, round(k/2))*gamPre ones(1, k-round(k/2))*gamPost];

betaStart = 0; betaEnd = 1;
gamStart = -1; gamEnd = 2;
theta_true(1,:) = linspace(betaStart, betaEnd, k);
theta_true(2,:) = linspace(gamStart, gamEnd, k);

lamSeq = exp(X_lam'.*theta_true(1,:));
nuSeq = exp(G_nu'.*theta_true(2,:));

spk_vec = com_rnd(lamSeq, nuSeq);
[mean_true,var_true] = getMeanVar(lamSeq, nuSeq);

subplot(1,3,1)
plot(lamSeq)
title('\lambda')
subplot(1,3,2)
plot(nuSeq)
title('\nu')
subplot(1,3,3)
hold on
plot(spk_vec)
plot(mean_true, 'LineWidth', 2)
hold off
title('spks')

%%
Q = diag([1e-3 1e-3]);
windType = 'forward';
F = diag([1 1]);

theta0 = theta_true(:, 1);
W0 = eye(2)*1e-2;

[theta_filt_exact, W_filt_exact] =...
    ppasmoo_compoisson_v2_window(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

theta_filt =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, 1, windType);

windSize = 5;
theta_filt_wind =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec,X_lam,G_nu,...
    W0,F,Q, windSize, windType);

gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, X_lam,G_nu, theta0, W0,...
    F, Q, spk_vec);
[theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_filt(:),1e-10,1000);
theta_newton = reshape(theta_newton_vec, [], k);

%%
figure;
subplot(1,2,1)
hold on
plot(theta_true(1,:))
plot(theta_filt_exact(1,:))
plot(theta_filt(1,:))
plot(theta_filt_wind(1,:))
plot(theta_newton(1,:))
hold off
title('\beta')
legend('true', 'smoother-exact', 'smoother', 'window-5','NR',...
    'Location','northwest')
subplot(1,2,2)
hold on
plot(theta_true(2,:))
plot(theta_filt_exact(2,:))
plot(theta_filt(2,:))
plot(theta_filt_wind(2,:))
plot(theta_newton(2,:))
title('\gamma')
hold off

figure;
subplot(1,2,1)
hold on
plot(exp(X_lam'.*theta_true(1,:)))
plot(exp(X_lam'.*theta_filt_exact(1,:)))
plot(exp(X_lam'.*theta_filt(1,:)))
plot(exp(X_lam'.*theta_filt_wind(1,:)))
plot(exp(X_lam'.*theta_newton(1,:)))
hold off
title('\lambda')
legend('true', 'smoother-exact', 'smoother', 'window-5','NR',...
    'Location','best')
subplot(1,2,2)
hold on
plot(exp(G_nu'.*theta_true(2,:)))
plot(exp(G_nu'.*theta_filt_exact(2,:)))
plot(exp(G_nu'.*theta_filt(2,:)))
plot(exp(G_nu'.*theta_filt_wind(2,:)))
plot(exp(G_nu'.*theta_newton(2,:)))
title('\nu')
hold off

[est_mean_filt_exact,est_var_filt_exact]=...
    getMeanVar(exp(X_lam'.*theta_filt_exact(1,:)),exp(G_nu'.*theta_filt_exact(2,:)));
[est_mean_filt,est_var_filt]=...
    getMeanVar(exp(X_lam'.*theta_filt(1,:)),exp(G_nu'.*theta_filt(2,:)));
[est_mean_filt_wind,est_var_filt_wind]=...
    getMeanVar(exp(X_lam'.*theta_filt_wind(1,:)),exp(G_nu'.*theta_filt_wind(2,:)));
[est_mean_newton,est_var_newton]=...
    getMeanVar(exp(X_lam'.*theta_newton(1,:)),exp(G_nu'.*theta_newton(2,:)));

figure;
subplot(1,2,1)
hold on
plot(mean_true)
plot(est_mean_filt_exact)
plot(est_mean_filt)
plot(est_mean_filt_wind)
plot(est_mean_newton)
hold off
title('Mean')
legend('true', 'smoother-exact', 'smoother', 'window-5','NR',...
    'Location','best')
subplot(1,2,2)
hold on
plot(var_true./ mean_true)
plot(est_var_filt_exact./ est_mean_filt_exact)
plot(est_var_filt./ est_mean_filt)
plot(est_var_filt_wind./ est_mean_filt_wind)
plot(est_var_newton./ est_mean_newton)
title('FF')
hold off

%% plot grid of llhd
clear all; close all; clc;
rng(3)

betaStart = 0; 
gamStart = -1;

nGrid_nu = 5;
nGrid_lam = 10;
betaRange = linspace(0,3,nGrid_lam);
gamRange = linspace(0,3,nGrid_nu);

testLlhd_filt_exact = zeros(nGrid_nu, nGrid_lam);
testLlhd_filt = zeros(nGrid_nu, nGrid_lam);
testLlhd_filt_wind = zeros(nGrid_nu, nGrid_lam);
testLlhd_newton = zeros(nGrid_nu, nGrid_lam);

for i = 1:nGrid_lam
    for j = 1:nGrid_nu
        llhd_mean_tmp = testLlhdCalc(betaStart,betaStart + betaRange(i),...
            gamStart,gamStart + gamRange(j),200,10);
        
        testLlhd_filt_exact(j,i) = llhd_mean_tmp(1);
        testLlhd_filt(j,i) = llhd_mean_tmp(2);
        testLlhd_filt_wind(j,i) = llhd_mean_tmp(3);
        testLlhd_newton(j,i) = llhd_mean_tmp(4);
        
    end
end

% compare to fisher scoring smoother
cLim_all = [min([testLlhd_filt_exact(:) - testLlhd_filt(:);...
    testLlhd_filt_wind(:) - testLlhd_filt(:);...
    testLlhd_newton(:) - testLlhd_filt(:)])...
    max([testLlhd_filt_exact(:) - testLlhd_filt(:);...
    testLlhd_filt_wind(:) - testLlhd_filt(:);...
    testLlhd_newton(:) - testLlhd_filt(:)])];
subplot(1,3,1)
imagesc(betaRange, gamRange,testLlhd_filt_exact - testLlhd_filt)
ylabel('range of \gamma')
title('exact - fisher')
colorbar()
set(gca,'CLim',cLim_all)
subplot(1,3,2)
imagesc(betaRange, gamRange,testLlhd_filt_wind - testLlhd_filt)
title('window 5 - fisher')
colorbar()
set(gca,'CLim',cLim_all)
subplot(1,3,3)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt)
xlabel('range of \beta')
title('newton - fisher')
colorbar()
set(gca,'CLim',cLim_all)

% newton vs. window 5
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt_wind)
title('newton - window 5')
colorbar()










