addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%% true underlying mean & FF
nknots = 10;
x0 = linspace(0,1,100);
basX = getCubicBSplineBasis(x0,nknots,false);

T = 100;
dt = 1;
kStep = T/dt;

gam = repmat(linspace(-1, 1, kStep), length(x0), 1);
nu = exp(gam);
nuSing = nu(1, :);

%
beta = zeros(size(basX, 2), kStep);
basMean = 3;
logLamBas = nuSing.*log(basMean + (nuSing - 1)./ (2*nuSing));

targetMean = linspace(6, 15, kStep) - basMean;
logLam = nuSing.*log(targetMean + (nuSing - 1)./ (2*nuSing));
weightSum = logLam/max(max(basX(:,2:end)));
weightKnots = 3;

weightBas = getCubicBSplineBasis(linspace(0,1,kStep),weightKnots,false);
plot(weightBas(:,2:end))
weight = weightBas(:,2:end)./repmat(sum(weightBas(:,2:end), 2), 1, weightKnots);

beta(1,:) = logLamBas;
beta(6,:) = weight(:,1).*weightSum';
beta(7,:) = weight(:,2).*weightSum';
beta(8,:) = weight(:,3).*weightSum';

lam = exp(basX*beta);

CMP_mean = zeros(size(lam));
CMP_var = zeros(size(lam));

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        logcum_app = logsum_calc(lam(m,n), nu(m,n), 100);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        
        CMP_mean(m,n) = exp(log_A - log_Z);
        CMP_var(m,n) = exp(log_B - log_Z) - CMP_mean(m,n)^2;
        
    end
end


%%
spk = zeros(size(lam));
rng(6)

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        spk(m,n) = com_rnd(lam(m,n), nu(m,n), 1);
        
    end
end

%% plot

figure(1)
subplot(1,3,1)
imagesc(dt:dt:T, x0,spk)
title('obs. spike counts')
ylabel('direction(rad)')
colorbar()
subplot(1,3,2)
imagesc(dt:dt:T, x0, CMP_mean)
title('Mean Firing Rate')
colorbar()
subplot(1,3,3)
imagesc(dt:dt:T, x0,CMP_var./CMP_mean)
title('Fano Factor')
xlabel('T')
colorbar()

t1=round(kStep/5);
t2=round(kStep*4/5);

figure(2)
subplot(1,2,1)
plot(CMP_mean(:,t1),'b')
hold on
plot(spk(:,t1),'b.')
plot(CMP_mean(:,t2),'r')
plot(spk(:,t2),'r.')
hold off
xlabel('Direction')
ylabel('Mean')

subplot(1,2,2)
plot(CMP_var(:,t1)./CMP_mean(:,t1),'b')
hold on
plot(CMP_var(:,t2)./CMP_mean(:,t2),'r')
hold off
legend({"t1 = "+t1,"t2 = "+t2})
xlabel('Direction')
ylabel('Fano Factor')

%% adaptive filtering
% still use single observation each step
% to match model fitting in the application part...
basX_trans = repmat(basX, kStep, 1);
spk_vec = spk(:);

b0 = glmfit(basX_trans(1:length(x0),:),spk_vec(1:length(x0)),'poisson','constant','off');

[theta_POI,W_POI, ~, lam_POI] =...
ppasmoo_poissexp(spk(:),basX_trans, b0,eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));
lam_POI_all = exp(basX*theta_POI);

writematrix(spk_vec(1:length(x0)), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(basX_trans(1:length(x0),:),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(ones(length(x0), 1),...
    'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\G.csv')

windType = 'forward';
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

[theta_CMP,W_CMP,~,~,~,~,~,~,...
    lam_CMP,nu_CMP,logZ_CMP] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk(:)',basX_trans, ones(length(x0)*kStep, 1),...
    eye(length(theta0)),eye(length(theta0)),1e-4*eye(length(theta0)), 20, windType);

CMP_mean_fit = 0*lam_CMP;
CMP_var_fit = 0*lam_CMP;

for m = 1:size(theta_CMP, 2)
    logcum_app  = logsum_calc(lam_CMP(m), nu_CMP(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    CMP_mean_fit(m) = exp(log_A - log_Z);
    CMP_var_fit(m) = exp(log_B - log_Z) - CMP_mean_fit(m)^2;
    
end

% lam_CMP_all = exp(basX*theta_CMP(1:(end-1), :));
% nu_CMP_all = exp(theta_CMP(end,:));
% 
% CMP_mean_fit_all = 0*lam_CMP_all;
% CMP_var_fit_all = 0*lam_CMP_all;
% 
% for m = 1:length(x0)
%     for n = 1:size(theta_CMP, 2)
%         logcum_app  = logsum_calc(lam_CMP_all(m, n), nu_CMP_all(n), 1000);
%         log_Z = logcum_app(1);
%         log_A = logcum_app(2);
%         log_B = logcum_app(3);
%         
%         CMP_mean_fit_all(m, n) = exp(log_A - log_Z);
%         CMP_var_fit_all(m, n) = exp(log_B - log_Z) - CMP_mean_fit_all(m, n)^2;
%         
%     end
% end



%% fitting plot

% % version 1: plot full fitting results
% % expand the observations
% % T_all = length(x0)*T 
% figure(3)
% subplot(2,3,1)
% imagesc(repelem(CMP_mean, 1, length(x0)))
% cLim_mean = caxis;
% % colorbar()
% title('expanded true mean')
% subplot(2,3,2)
% imagesc(lam_POI_all)
% % colorbar()
% set(gca,'CLim',cLim_mean)
% title('expanded Poisson mean')
% subplot(2,3,3)
% imagesc(CMP_mean_fit_all)
% colorbar()
% set(gca,'CLim',cLim_mean)
% title('expanded CMP mean')
% subplot(2,3,4)
% imagesc(repelem(CMP_var./CMP_mean, 1, length(x0)))
% cLim_ff = caxis;
% % colorbar()
% title('expanded true FF')
% subplot(2,3,5)
% imagesc(ones(size(lam_POI_all)))
% % colorbar()
% set(gca,'CLim',cLim_ff)
% title('expanded Poisson FF')
% subplot(2,3,6)
% imagesc(CMP_var_fit_all./CMP_mean_fit_all)
% colorbar()
% set(gca,'CLim',cLim_ff)
% title('expanded CMP FF')
% 
% 
% % version 2: use the last fitting within each step
% figure(4)
% subplot(2,3,1)
% imagesc(CMP_mean)
% cLim_mean = caxis;
% % colorbar()
% title('true mean')
% subplot(2,3,2)
% imagesc(lam_POI_all(:, length(x0):length(x0):end))
% % colorbar()
% set(gca,'CLim',cLim_mean)
% title('Poisson mean: last')
% subplot(2,3,3)
% imagesc(CMP_mean_fit_all(:, length(x0):length(x0):end))
% colorbar()
% set(gca,'CLim',cLim_mean)
% title('CMP mean: last')
% subplot(2,3,4)
% imagesc(CMP_var./CMP_mean)
% cLim_ff = caxis;
% % colorbar()
% title('true FF')
% subplot(2,3,5)
% imagesc(ones(size(CMP_mean)))
% % colorbar()
% set(gca,'CLim',cLim_ff)
% title('Poisson FF: last')
% subplot(2,3,6)
% ff_fit_all = CMP_var_fit_all./CMP_mean_fit_all;
% imagesc(ff_fit_all(:, length(x0):length(x0):end))
% colorbar()
% set(gca,'CLim',cLim_ff)
% title('CMP FF: last')


% version 3: plot each point seperately
figure(5)
subplot(2,3,1)
imagesc(CMP_mean)
cLim_mean = caxis;
% colorbar()
title('true mean')
subplot(2,3,2)
imagesc(reshape(lam_POI,length(x0),[]))
% colorbar()
set(gca,'CLim',cLim_mean)
title('Poisson mean: point')
subplot(2,3,3)
imagesc(reshape(CMP_mean_fit,length(x0),[]))
colorbar()
set(gca,'CLim',cLim_mean)
title('CMP mean: point')
subplot(2,3,4)
imagesc(CMP_var./CMP_mean)
cLim_ff = caxis;
% colorbar()
title('true FF')
subplot(2,3,5)
imagesc(ones(size(CMP_mean)))
% colorbar()
set(gca,'CLim',cLim_ff)
title('Poisson FF: point')
subplot(2,3,6)
imagesc(reshape(CMP_var_fit,length(x0),[])./reshape(CMP_mean_fit,length(x0),[]))
colorbar()
set(gca,'CLim',cLim_ff)
title('CMP FF: point')


