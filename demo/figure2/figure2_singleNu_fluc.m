addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%% true underlying mean & FF
nknots = 5;
x0 = linspace(0,1,100);
basX = getCubicBSplineBasis(x0,nknots,false);

T = 100;
dt = 1;
kStep = T/dt;

bits = sign(mod(1:24,2)-0.5);
nu_fluc = interp1(linspace(0,1,24),randsample(bits,24)+randn(1,24)/5,linspace(0,1,T),'spline');

gam = repmat(nu_fluc'/4, 1, kStep)';
nu = exp(gam);
nuSing = nu(1, :);

%
beta = zeros(size(basX, 2), kStep);
% basMean = 2;
% logLamBas = nuSing.*log(basMean + (nuSing - 1)./ (2*nuSing));

% targetMean = linspace(5, 5, kStep) - basMean;
% logLam = nuSing.*log(targetMean + (nuSing - 1)./ (2*nuSing));
% weightSum = logLam/max(max(basX(:,2:end)));
% weightKnots = 3;

% weightBas = getCubicBSplineBasis(linspace(0,1,kStep),weightKnots,false);
% % plot(weightBas(:,2:end))
% weight = weightBas(:,2:end)./repmat(sum(weightBas(:,2:end), 2), 1, weightKnots);

beta(1,:) = 4;
beta(4,:) = 10;

% lam = exp(nu.*log(exp(basX*beta + (nu - 1)./(2.*nu))));

lam = log(exp(basX*beta + (nu - 1)./(2.*nu))+1).^nu;

% lam = repmat(lam(:,1),1,kStep);

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
% basX_trans = repmat([basX(:,1) basX(:,2:end)*beta(2:end,1)],kStep,1);
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

Q = 1e-3*eye(length(theta0));
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec', basX_trans, ones(Tall, 1),...
    eye(length(theta0)),eye(length(theta0)),Q);


theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = -20;
QUB = 1;
Q0 = -6*ones(1, min(2, size(basX_trans, 2))+ min(2, size(ones(Tall, 1), 2)));
Q0 = [-10 -5 -5];
DiffMinChange = 0.01;
DiffMaxChange = 1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(exp(Q), theta01, spk_vec',basX_trans,ones(Tall, 1),...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);
Qopt = exp(Qopt);
%
Qopt = [10e-6 2*10e-4 2*10e-4];
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

% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure1';
% cd(plotFolder)

angle = 180*(x0-min(x0))/range(x0);

[maxt1,idt1] = max(CMP_mean(:,t1));
[maxt2,idt2] = max(CMP_mean(:,t2));

FFtrue = CMP_var./CMP_mean;
[~,t1]=max(FFtrue(idt1,:));
[~,t2]=min(FFtrue(idt1,:));

MFR = figure(1);
imagesc(1:dt:T, angle, CMP_mean)
% title('Mean Firing Rate')
xlabel('Trial')
ylabel('Orientation (degree)')
yline(angle(idt1), 'b--', 'LineWidth', 2);
yline(angle(idt2), 'r--', 'LineWidth', 2);
xline(t1, 'b--', 'LineWidth', 2);
xline(t2, 'r--', 'LineWidth', 2);
colormap(flipud(gray(256)));
colorbar;
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

% set(MFR,'PaperUnits','inches','PaperPosition',[0 0 5 3])
% saveas(MFR, '1_MFR.svg')
% saveas(MFR, '1_MFR.png')

FF = figure(2);
imagesc(1:dt:T, angle,CMP_var./CMP_mean)
% title('Fano Factor')
xlabel('Trial')
ylabel('Orientation (degree)')
yline(angle(idt1), 'b--', 'LineWidth', 2);
yline(angle(idt2), 'r--', 'LineWidth', 2);
xline(t1, 'b--', 'LineWidth', 2);
xline(t2, 'r--', 'LineWidth', 2);
colormap(flipud(gray(256)));
colorbar;
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

% set(FF,'PaperUnits','inches','PaperPosition',[0 0 5 3])
% saveas(FF, '2_FF.svg')
% saveas(FF, '2_FF.png')



FR2 = figure(3);
plot(angle, CMP_mean(:,t1),'b', 'LineWidth', 2)
hold on
plot(angle, CMP_mean(:,t2),'r', 'LineWidth', 2)
plot(angle, CMP_mean_fit_trans(:,t1),'b--', 'LineWidth', 2)
plot(angle, CMP_mean_fit_trans(:,t2),'r--', 'LineWidth', 2)
plot(angle, spk(:,t1),'b.')
plot(angle, spk(:,t2),'r.')
plot(angle(idt1), maxt1, 'o', 'Color', 'b',...
    'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)
plot(angle(idt2), maxt2, 'o', 'Color', 'r',...
    'LineWidth', 2, 'markerfacecolor', 'r', 'MarkerSize',5)
hold off
xlabel('Orientation (degree)')
ylabel('Mean Firing Rate')
xlim([0 180])
% legend({"true: t_1 = "+t1,"true: t_2 = "+t2,"fit: t_1 = "+t1,"fit: t_2 = "+t2})
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

% set(FR2,'PaperUnits','inches','PaperPosition',[0 0 5 3])
% saveas(FR2, '3_FR2.svg')
% saveas(FR2, '3_FR2.png')

FF = figure(4);
plot(CMP_var(idt1,:)./CMP_mean(idt1,:),'b', 'LineWidth', 2)
hold on
plot(CMP_var(idt2,:)./CMP_mean(idt2,:),'r', 'LineWidth', 2)
plot(CMP_ff_fit_trans(idt1,:), 'b--', 'LineWidth', 2)
plot(CMP_ff_fit_trans(idt2,:), 'r--', 'LineWidth', 2)
hold off
% legend({"true: pos_1 = "+round(x0(idt1), 3),"true: pos_2 = "+round(x0(idt2), 3),...
%     "fit: pos_1 = "+round(x0(idt1), 3),"fit: pos_2 = "+round(x0(idt2), 3)})
xlabel('Trial')
ylabel('Fano Factor')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

% set(FF,'PaperUnits','inches','PaperPosition',[0 0 5 3])
% saveas(FF, '4_FF.svg')
% saveas(FF, '4_FF.png')


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

