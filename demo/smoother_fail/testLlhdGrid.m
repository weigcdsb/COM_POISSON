addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%% plot grid of llhd: linear
clear all; close all; clc;
rng(1)

betaStart = 0; 
% gamStart = -1;

% nGrid_nu = 10;
% nGrid_lam = 10;
% betaRange = linspace(0,3,nGrid_lam);
% gamRange = linspace(1,3,nGrid_nu);

% nu constant:
gamStart = .5;
nGrid_lam = 10;
nGrid_nu = 1;
betaRange = linspace(0,4,nGrid_lam);
gamRange = linspace(0,0,nGrid_nu);


testLlhd_filt_exact = zeros(nGrid_nu, nGrid_lam);
testLlhd_filt = zeros(nGrid_nu, nGrid_lam);
testLlhd_filt_wind = zeros(nGrid_nu, nGrid_lam);
testLlhd_newton = zeros(nGrid_nu, nGrid_lam);

% arbitrary window size
% windSize = 5;
% for i = 1:nGrid_lam
%     for j = 1:nGrid_nu
%         llhd_mean_tmp = testLlhdCalc_linear(betaStart,betaStart + betaRange(i),...
%             gamStart,gamStart + gamRange(j),200,10, windSize);
%         
%         testLlhd_filt_exact(j,i) = llhd_mean_tmp(1);
%         testLlhd_filt(j,i) = llhd_mean_tmp(2);
%         testLlhd_filt_wind(j,i) = llhd_mean_tmp(3);
%         testLlhd_newton(j,i) = llhd_mean_tmp(4);
%         
%     end
% end

% windSize select by forward chaining
windSize_opt = zeros(nGrid_nu, nGrid_lam);

nRep = 100;
for k = 1:nRep
    for i = 1:nGrid_lam
        for j = 1:nGrid_nu
            [llhd_mean_tmp, windSize_tmp]= testLlhdCalc_linear_CV(betaStart,betaStart + betaRange(i),...
                gamStart,gamStart + gamRange(j),200,1);
            
            windSize_opt(j,i) = windSize_opt(j,i) + windSize_tmp;
            testLlhd_filt_exact(j,i) = testLlhd_filt_exact(j,i) + llhd_mean_tmp(1);
            testLlhd_filt(j,i) = testLlhd_filt(j,i) + llhd_mean_tmp(2);
            testLlhd_filt_wind(j,i) = testLlhd_filt_wind(j,i) + llhd_mean_tmp(3);
            testLlhd_newton(j,i) = testLlhd_newton(j,i) + llhd_mean_tmp(4);
        end
    end
end

windSize_opt = windSize_opt/nRep;
testLlhd_filt_exact = testLlhd_filt_exact/nRep;
testLlhd_filt = testLlhd_filt/nRep;
testLlhd_filt_wind = testLlhd_filt_wind/nRep;
testLlhd_newton = testLlhd_newton/nRep;

%% plot grid of llhd: constant
% clear all; close all; clc;
% rng(3)
% 
% nGrid_nu = 10;
% nGrid_lam = 10;
% betaRange = linspace(0.5,2.5,nGrid_lam);
% gamRange = linspace(-.5,2.5,nGrid_nu);
% 
% testLlhd_filt_exact = zeros(nGrid_nu, nGrid_lam);
% testLlhd_filt = zeros(nGrid_nu, nGrid_lam);
% testLlhd_filt_wind = zeros(nGrid_nu, nGrid_lam);
% testLlhd_newton = zeros(nGrid_nu, nGrid_lam);
% 
% windSize_opt = zeros(nGrid_nu, nGrid_lam);
% for i = 1:nGrid_lam
%     for j = 1:nGrid_nu
%         [llhd_mean_tmp, windSize_opt(j,i)]= testLlhdCalc_constant_CV(betaRange(i),...
%             gamRange(j),200,10);
%         
%         testLlhd_filt_exact(j,i) = llhd_mean_tmp(1);
%         testLlhd_filt(j,i) = llhd_mean_tmp(2);
%         testLlhd_filt_wind(j,i) = llhd_mean_tmp(3);
%         testLlhd_newton(j,i) = llhd_mean_tmp(4);
%         
%     end
% end


%% plot
% compare to fisher scoring smoother
cLim_all = [min([testLlhd_filt_exact(:) - testLlhd_filt(:);...
    testLlhd_filt_wind(:) - testLlhd_filt(:);...
    testLlhd_newton(:) - testLlhd_filt(:)])...
    max([testLlhd_filt_exact(:) - testLlhd_filt(:);...
    testLlhd_filt_wind(:) - testLlhd_filt(:);...
    testLlhd_newton(:) - testLlhd_filt(:)])];
figure;
subplot(1,3,1)
imagesc(betaRange, gamRange,testLlhd_filt_exact - testLlhd_filt)
ylabel("range of \gamma, start from " + gamStart)
title('exact - fisher')
colorbar()
set(gca,'CLim',cLim_all)
subplot(1,3,2)
imagesc(betaRange, gamRange,testLlhd_filt_wind - testLlhd_filt)
% title("window " + windSize + " - fisher")
title('window - fisher')
colorbar()
set(gca,'CLim',cLim_all)
subplot(1,3,3)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt)
xlabel("range of \beta, start from" + betaStart)
title('newton - fisher')
colorbar()
set(gca,'CLim',cLim_all)

% to show negative
figure;
subplot(1,3,1)
imagesc(betaRange, gamRange,testLlhd_filt_exact - testLlhd_filt)
ylabel("range of \gamma, start from " + gamStart)
title('exact - fisher')
colorbar()
subplot(1,3,2)
imagesc(betaRange, gamRange,testLlhd_filt_wind - testLlhd_filt)
% title("window " + windSize + " - fisher")
title('window - fisher')
colorbar()
subplot(1,3,3)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt)
xlabel("range of \beta, start from" + betaStart)
title('newton - fisher')
colorbar()

% newton vs. window
figure;
subplot(1,2,1)
imagesc(betaRange, gamRange,(testLlhd_newton - testLlhd_filt_wind)>0)
title('newton - window: >0 ?')
ylabel("range of \gamma, start from " + gamStart)
colorbar()
subplot(1,2,2)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt_wind)
% title('newton - window 5')
title('newton - window: value')
xlabel("range of \beta, start from" + betaStart)
colorbar()

% selected window size
figure;
imagesc(betaRange, gamRange,windSize_opt)
title('selected window size')
ylabel("range of \gamma, start from " + gamStart)
xlabel("range of \beta, start from" + betaStart)
colorbar()