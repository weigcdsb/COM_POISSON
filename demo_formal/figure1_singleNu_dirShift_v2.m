addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%% true underlying mean & FF
x0 = linspace(0,1,100);
T = 200;
t = linspace(0,1,T);
x0 = linspace(0,2*pi,100);

Xplot = repmat(x0',1,T);
Tplot = repmat(t,length(x0),1);
[lam,nu] = cmpDriftGenerator(8,Xplot,Tplot);

CMP_mean = zeros(size(lam));
CMP_var = zeros(size(lam));

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        logcum_app = logsum_calc(lam(m,n), nu(m,n), 1000);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        
        CMP_mean(m,n) = exp(log_A - log_Z);
        CMP_var(m,n) = exp(log_B - log_Z) - CMP_mean(m,n)^2;
        
    end
end

% plot ground-truth

figure(1)
subplot(1,2,1)
imagesc(1:T, x0, CMP_mean)
title('Mean Firing Rate')
colorbar()
subplot(1,2,2)
imagesc(1:T, x0,CMP_var./CMP_mean)
title('Fano Factor')
xlabel('T')
colorbar()

%% sample from the model...
x0_deg = x0*180/pi;

ndir = 12;
xsamp0 = linspace(0,2*pi-2*pi/ndir,ndir);
xsamp=[];
for i=1:T
   xsamp = [xsamp xsamp0(randperm(length(xsamp0)))]; 
end
tsamp = linspace(0,1,length(xsamp));
[lamsamp,nusamp] = cmpDriftGenerator(8,xsamp,tsamp);


spk = zeros(size(lamsamp));
for m = 1:length(lamsamp)
    spk(m) = com_rnd(lamsamp(m), nusamp(m), 1);
end


t1=20; t2=120;
sampid1 = abs(tsamp-t1/T)<0.01;
sampid2 = abs(tsamp-t2/T)<0.01;

figure(2)
subplot(1,2,1)
plot(x0_deg,CMP_mean(:,t1),'b')
hold on
plot(xsamp(sampid1)*180/pi+randn(1,sum(sampid1))*5-2.5,spk(sampid1),'b.')
plot(x0_deg,CMP_mean(:,t2),'r')
plot(xsamp(sampid2)*180/pi+randn(1,sum(sampid2))*5-2.5,spk(sampid2),'r.')
hold off
xlabel('Direction [deg]')
ylabel('Mean')

subplot(1,2,2)
plot(x0_deg,CMP_var(:,t1)./CMP_mean(:,t1),'b')
hold on
plot(x0_deg,CMP_var(:,t2)./CMP_mean(:,t2),'r')
hold off
legend({"t1 = "+t1,"t2 = "+t2})
xlabel('Direction')
ylabel('Fano Factor')

%%

nknots = 3;

% x0 = linspace(0,1,100);
% basX = getCubicBSplineBasis(x0,nknots,false);

% for periodic boundary condition

basX = getCubicBSplineBasis(x0,nknots,true);
bassamp = getCubicBSplineBasis(xsamp,nknots,true);

%% adaptive filtering

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';

% usr_dir = 'C:\Users\ian';
% r_path = 'C:\Program Files\R\R-4.0.2\bin';


b0 = glmfit(bassamp(1:100,:),spk(1:100)','poisson','constant','off');

[theta_POI,W_POI, ~, lam_POI] =...
ppasmoo_poissexp(spk(:),bassamp, b0,eye(length(b0)),eye(length(b0)),1e-3*eye(length(b0)));
lam_POI_plot = exp(basX*theta_POI(:,1:ndir:end));

writematrix(spk(1:200)', [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(bassamp(1:200,:),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(ones(200, 1),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

windType = 'forward';
RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],...
    r_path);

theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);

Qcmp = 1e-3*eye(length(theta0));
Qcmp(end,end)=1e-3;
[theta_CMP,W_CMP,~,~,~,~,~,~,...
    lam_CMP,nu_CMP,logZ_CMP] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk(:)',bassamp, ones(size(spk))',...
    eye(length(theta0)),eye(length(theta0)),Qcmp, 50, windType);

lam_CMP_plot = exp(basX*theta_CMP(1:end-1,1:ndir:end));
nu_CMP_plot = exp(theta_CMP(end,1:ndir:end));

CMP_mean_fit = 0*lam_CMP_plot;
CMP_var_fit = 0*lam_CMP_plot;

for m = 1:size(lam_CMP_plot, 1)
    for n=1:size(lam_CMP_plot, 2)
        logcum_app  = logsum_calc(lam_CMP_plot(m,n), nu_CMP_plot(n), 500);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
    
        CMP_mean_fit(m,n) = exp(log_A - log_Z);
        CMP_var_fit(m,n) = exp(log_B - log_Z) - CMP_mean_fit(m,n)^2;
    end
end

%% fitting plot

% version 3: plot each point seperately
figure(5)
subplot(2,3,1)
imagesc(CMP_mean)
cLim_mean = caxis;
colorbar()
title('true mean')
subplot(2,3,2)
imagesc(lam_POI_plot)
colorbar()
set(gca,'CLim',cLim_mean)
title('Poisson mean: point')
subplot(2,3,3)
imagesc(CMP_mean_fit)
colorbar()
set(gca,'CLim',cLim_mean)
title('CMP mean: point')

subplot(2,3,4)
imagesc(CMP_var./CMP_mean)
cLim_ff = caxis;
colorbar()
title('true FF')
subplot(2,3,5)
imagesc(ones(size(CMP_mean)))
colorbar()
set(gca,'CLim',cLim_ff)
title('Poisson FF: point')
subplot(2,3,6)
imagesc(CMP_var_fit./CMP_mean_fit)
colorbar()
set(gca,'CLim',cLim_ff)
title('CMP FF: point')

%% diagnosis: KS-test?

spkFit_true = zeros(size(lam));
spkFit_CMP = zeros(size(lam_CMP_plot));
spkFit_POI = poissrnd(lam_POI_plot);
for m = 1:size(lam_CMP_plot, 1)
    for n = 1:size(lam_CMP_plot, 2)
        spkFit_true(m,n) = com_rnd(lam(m,n), nu(m,n), 1);
        spkFit_CMP(m,n) = com_rnd(lam_CMP_plot(m,n), nu_CMP_plot(n), 1);
    end
end

% compare 2 fit
% keep Qcmp = 1e-3

[~,p0] = kstest2(spkFit_POI(:),spkFit_CMP(:));
p0 
% = 3.5605e-09 Qpoi = 1e-1
% = 2.5729e-05 Qpoi = 1e-2
% = 1.9312e-05 Qpoi = 1e-3
% = 1.7931e-07 Qpoi = 1e-4

% compare to samples
[~,p11] = kstest2(spkFit_POI(:),spk(:));
[~,p12] = kstest2(spkFit_CMP(:),spk(:));
p11 
% = 0.0612 Qpoi = 1e-1
% = 0.1192 Qpoi = 1e-2
% = 0.4249 Qpoi = 1e-3
% = 0.1741 Qpoi = 1e-4

p12 % = 0.9914

% compare to full samples
[~,p21] = kstest2(spkFit_POI(:),spkFit_true(:));
[~,p22] = kstest2(spkFit_CMP(:),spkFit_true(:));

p21 
% = 2.4688e-06 Qpoi = 1e-1
% = 1.5164e-05 Qpoi = 1e-2
% = 0.0016 Qpoi = 1e-3
% = 2.3392e-05 Qpoi = 1e-4

p22 % = 0.6172
