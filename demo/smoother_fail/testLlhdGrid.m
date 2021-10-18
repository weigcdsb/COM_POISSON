addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%% plot grid of llhd: linear
clear all; close all; clc;
rng(2)

% betaStart = 0; 
% gamStart = -1;
% 
% nGrid_nu = 10;
% nGrid_lam = 10;
% betaRange = linspace(0,3,nGrid_lam);
% gamRange = linspace(1,3,nGrid_nu);

% nu constant:
betaStart = 0;
gamStart = 0;
nGrid_lam = 10;
nGrid_nu = 1;
betaRange = linspace(0,3,nGrid_lam);
gamRange = linspace(0,0,nGrid_nu);

% beta constant
% betaStart = -0.5;
% gamStart = -1;
% nGrid_lam = 1;
% nGrid_nu = 10;
% betaRange = linspace(0,0,nGrid_lam);
% gamRange = linspace(0,3,nGrid_nu);


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

nRep = 100;
windSize_opt_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_exact_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_wind_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_newton_all = zeros(nGrid_nu, nGrid_lam, nRep);

for k = 1:nRep
    disp(k)
    for i = 1:nGrid_lam
        for j = 1:nGrid_nu
            
            nanFlag = true;
            while(nanFlag)
                [llhd_mean_tmp, windSize_opt_all(j,i,k)]=...
                    testLlhdCalc_linear_CV(betaStart,betaStart + betaRange(i),...
                    gamStart,gamStart + gamRange(j),200,1);
                
                if(sum(isnan(llhd_mean_tmp)) == 0)
                    nanFlag = false;
                end
            end
            
            testLlhd_filt_exact_all(j,i,k) = llhd_mean_tmp(1);
            testLlhd_filt_all(j,i,k) = llhd_mean_tmp(2);
            testLlhd_filt_wind_all(j,i,k) = llhd_mean_tmp(3);
            testLlhd_newton_all(j,i,k) = llhd_mean_tmp(4);
        end
    end
end

windSize_opt = mean(windSize_opt_all, 3);
testLlhd_filt_exact = mean(testLlhd_filt_exact_all,3);
testLlhd_filt =  mean(testLlhd_filt_all, 3);
testLlhd_filt_wind =  mean(testLlhd_filt_wind_all, 3);
testLlhd_newton =  mean(testLlhd_newton_all, 3);

%% plot grid of llhd: constant
clear all; close all; clc;
rng(3)

nGrid_nu = 10;
nGrid_lam = 10;
betaRange = linspace(0.5,2.5,nGrid_lam);
gamRange = linspace(-.5,2.5,nGrid_nu);

nRep = 100;
windSize_opt_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_exact_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_filt_wind_all = zeros(nGrid_nu, nGrid_lam, nRep);
testLlhd_newton_all = zeros(nGrid_nu, nGrid_lam, nRep);

for k = 1:nRep
    disp(k)
    for i = 1:nGrid_lam
        for j = 1:nGrid_nu
            
            nanFlag = true;
            while(nanFlag)
                [llhd_mean_tmp, windSize_opt_all(j,i,k)]=...
                    testLlhdCalc_constant_CV(betaRange(i),gamRange(j),200,1);
                
                
                if(sum(isnan(llhd_mean_tmp)) == 0)
                    nanFlag = false;
                end
            end
            
            testLlhd_filt_exact_all(j,i,k) = llhd_mean_tmp(1);
            testLlhd_filt_all(j,i,k) = llhd_mean_tmp(2);
            testLlhd_filt_wind_all(j,i,k) = llhd_mean_tmp(3);
            testLlhd_newton_all(j,i,k) = llhd_mean_tmp(4);
        end
    end
end

windSize_opt = mean(windSize_opt_all, 3);
testLlhd_filt_exact = mean(testLlhd_filt_exact_all,3);
testLlhd_filt =  mean(testLlhd_filt_all, 3);
testLlhd_filt_wind =  mean(testLlhd_filt_wind_all, 3);
testLlhd_newton =  mean(testLlhd_newton_all, 3);


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
% ylabel("range of \gamma, start from " + gamStart)
ylabel("\gamma")
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
% xlabel("range of \beta, start from" + betaStart)
xlabel("\beta")
title('newton - fisher')
colorbar()
set(gca,'CLim',cLim_all)

% to show negative
figure;
subplot(1,3,1)
imagesc(betaRange, gamRange,testLlhd_filt_exact - testLlhd_filt)
% ylabel("range of \gamma, start from " + gamStart)
ylabel("\gamma")
title('exact - fisher')
colorbar()
subplot(1,3,2)
imagesc(betaRange, gamRange,testLlhd_filt_wind - testLlhd_filt)
% title("window " + windSize + " - fisher")
title('window - fisher')
colorbar()
subplot(1,3,3)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt)
% xlabel("range of \beta, start from" + betaStart)
xlabel("\beta")
title('newton - fisher')
colorbar()

% newton vs. window
figure;
subplot(1,2,1)
imagesc(betaRange, gamRange,(testLlhd_newton - testLlhd_filt_wind)>0)
title('newton - window: >0 ?')
% ylabel("range of \gamma, start from " + gamStart)
ylabel("\gamma")
colorbar()
subplot(1,2,2)
imagesc(betaRange, gamRange,testLlhd_newton - testLlhd_filt_wind)
% title('newton - window 5')
title('newton - window: value')
% xlabel("range of \beta, start from" + betaStart)
xlabel("\beta")
colorbar()

% selected window size
figure;
imagesc(betaRange, gamRange,windSize_opt)
title('selected window size')
% ylabel("range of \gamma, start from " + gamStart)
ylabel("\gamma")
% xlabel("range of \beta, start from" + betaStart)
xlabel("\beta")
colorbar()


save('C:\Users\gaw19004\Desktop\COM_POI_data\compare_test_llhd\double_constant.mat')
