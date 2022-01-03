addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

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
basMean = 2;
logLamBas = nuSing.*log(basMean + (nuSing - 1)./ (2*nuSing));

targetMean = linspace(5, 10, kStep) - basMean;
logLam = nuSing.*log(targetMean + (nuSing - 1)./ (2*nuSing));
weightSum = logLam/max(max(basX(:,2:end)));
weightKnots = 3;

weightBas = getCubicBSplineBasis(linspace(0,1,kStep),weightKnots,false);
% plot(weightBas(:,2:end))
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
        [CMP_mean(m,n), CMP_var(m,n), ~, ~, ~, ~] = ...
            CMPmoment(lam(m,n), nu(m,n), 1000);
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

%% model fit
% still use single observation each step
% to match model fitting in the application part...
basX_trans = repmat(basX, kStep, 1);
spk_vec = spk(:);
Tall = length(x0)*kStep;

% b0 = glmfit(basX_trans(1:length(x0),:),spk_vec(1:length(x0)),'poisson','constant','off');
% [theta_POI,W_POI, ~, lam_POI] =...
% ppasmoo_poissexp(spk(:),basX_trans, b0,eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));
% lam_POI_all = exp(basX*theta_POI);

writematrix(spk_vec(1:length(x0)), [r_wd '\y.csv'])
writematrix(basX_trans(1:length(x0),:),[r_wd '\X.csv'])
writematrix(ones(length(x0), 1),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = 1e-4*eye(length(theta0));
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec', basX_trans, ones(Tall, 1),...
    eye(length(theta0)),eye(length(theta0)),Q);


theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(basX_trans, 2))+ min(2, size(ones(Tall, 1), 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(Q, theta01, spk_vec',basX_trans,ones(Tall, 1),...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt(1) Qopt(2)*ones(1, size(basX_trans, 2)-1)];
Q_nu = Qopt(3);
Qoptmatrix = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, basX_trans,ones(Tall, 1),...
    theta01, W01,eye(length(theta01)), Qoptmatrix, spk_vec');
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit = reshape(theta_newton_vec, [], Tall);

lam_fit = zeros(1, Tall);
nu_fit = zeros(1, Tall);
CMP_mean_fit = zeros(1, Tall);
CMP_var_fit = zeros(1, Tall);
logZ_fit = zeros(1,Tall);

for m = 1:Tall
    lam_fit(m) = exp(basX_trans(m,:)*theta_fit(1:(nknots+1), m));
    nu_fit(m) = exp(theta_fit((nknots+2):end, m));
    [CMP_mean_fit(m), CMP_var_fit(m), ~, ~, ~, logZ_fit(m)] = ...
            CMPmoment(lam_fit(m), nu_fit(m), 1000);
end

CMP_ff_fit = CMP_var_fit./CMP_mean_fit;

CMP_mean_fit_trans = reshape(CMP_mean_fit, [], kStep);
CMP_ff_fit_trans = reshape(CMP_ff_fit, [], kStep);

%% plot

figure(1)
% subplot(3,1,1)
% imagesc(1:dt:T, x0,spk)
% title('obs. spike counts')
% colorbar()
subplot(2,1,1)
imagesc(1:dt:T, x0, CMP_mean)
title('Mean Firing Rate')
ylabel('Direction(rad)')
colorbar()
subplot(2,1,2)
imagesc(1:dt:T, x0,CMP_var./CMP_mean)
title('Fano Factor')
xlabel('Trial')
colorbar()


t1=round(kStep/5);
t2=round(kStep*4/5);

figure(2)
plot(x0, CMP_mean(:,t1),'b')
[maxt1,idt1] = max(CMP_mean(:,t1));
hold on
plot(x0, CMP_mean(:,t2),'r')
[maxt2,idt2] = max(CMP_mean(:,t2));
plot(x0, CMP_mean_fit_trans(:,t1),'b--')
plot(x0, CMP_mean_fit_trans(:,t2),'r--')
plot(x0, spk(:,t1),'b.')
plot(x0, spk(:,t2),'r.')
plot(x0(idt1), maxt1, 'o', 'Color', 'b',...
    'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)
plot(x0(idt2), maxt2, 'o', 'Color', 'r',...
    'LineWidth', 2, 'markerfacecolor', 'r', 'MarkerSize',5)
hold off
xlabel('Direction(rad)')
ylabel('Mean')
legend({"true: t_1 = "+t1,"true: t_2 = "+t2,"fit: t_1 = "+t1,"fit: t_2 = "+t2})

figure(3)
plot(CMP_var(idt1,:)./CMP_mean(idt1,:),'b')
hold on
plot(CMP_var(idt2,:)./CMP_mean(idt2,:),'r')
plot(CMP_ff_fit_trans(idt1,:), 'b--')
plot(CMP_ff_fit_trans(idt2,:), 'r--')
hold off
legend({"true: pos_1 = "+round(x0(idt1), 3),"true: pos_2 = "+round(x0(idt2), 3),...
    "fit: pos_1 = "+round(x0(idt1), 3),"fit: pos_2 = "+round(x0(idt2), 3)})
xlabel('trial')
ylabel('Fano Factor')

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
% figure(5)
% subplot(2,3,1)
% imagesc(CMP_mean)
% cLim_mean = caxis;
% % colorbar()
% title('true mean')
% subplot(2,3,2)
% imagesc(reshape(lam_POI,length(x0),[]))
% % colorbar()
% set(gca,'CLim',cLim_mean)
% title('Poisson mean: point')
% subplot(2,3,3)
% imagesc(reshape(CMP_mean_fit,length(x0),[]))
% colorbar()
% set(gca,'CLim',cLim_mean)
% title('CMP mean: point')
% subplot(2,3,4)
% imagesc(CMP_var./CMP_mean)
% cLim_ff = caxis;
% % colorbar()
% title('true FF')
% subplot(2,3,5)
% imagesc(ones(size(CMP_mean)))
% % colorbar()
% set(gca,'CLim',cLim_ff)
% title('Poisson FF: point')
% subplot(2,3,6)
% imagesc(reshape(CMP_var_fit,length(x0),[])./reshape(CMP_mean_fit,length(x0),[]))
% colorbar()
% set(gca,'CLim',cLim_ff)
% title('CMP FF: point')

