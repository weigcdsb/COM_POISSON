addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%% true underlying mean & FF
rng(1)
nknots = 2;
x0 = linspace(0,1,50);
basX = getCubicBSplineBasis(x0,nknots,false);

T = 100;
dt = 1;
kStep = T/dt;

nu = 0.1*ones(size(x0,2),kStep);

mupf = interp1(linspace(1,T,10),randn(1,10)/10+0.5,linspace(1,T,T),'spline');
[tm,xm]=meshgrid(1:T,x0);

lam = exp(-(xm-mupf).^2/2/0.05);
imagesc(lam)
colorbar

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

% subplot(1,2,1)
% imagesc(CMP_mean)
% colorbar
% subplot(1,2,2)
% imagesc(spk)
% colorbar


%% split train & test
rng(111)
propTrain = 1/5;
splitIdx = spk*0;

for tt = 1:size(spk, 2)
    splitIdx(:,tt) = binornd(1,propTrain, 1, size(spk, 1));
end

spk_train = spk*nan;
spk_test = spk*nan;

spk_train(splitIdx == 1) = spk(splitIdx == 1);
spk_test(splitIdx == 0) = spk(splitIdx == 0);

% imagesc(spk_train)

% hold out within the place field
% rng(8)
% propTrain = 1/10;
% splitIdx = ones(size(spk));
% for tt = 1:size(spk, 2)
%     splitIdx(CMP_mean(:,tt) > 1,tt) = binornd(1,propTrain,...
%         1, sum(CMP_mean(:,tt) > 1));
% end
% 
% spk_train = spk*nan;
% spk_test = spk*nan;
% 
% spk_train(splitIdx == 1) = spk(splitIdx == 1);
% spk_test(splitIdx == 0) = spk(splitIdx == 0);


%% model fit
% still use single observation each step
% to match model fitting in the application part...
basX_trans = repmat(basX, kStep, 1);
% basX_trans = repmat([basX(:,1) basX(:,2:end)*beta(2:end,1)],kStep,1);
spk_vec = spk_train(:);
Tall = length(x0)*kStep;

nonNAidx = find(~isnan(spk_vec));
writematrix(spk_vec(nonNAidx(nonNAidx < length(x0))), [r_wd '\y.csv'])
writematrix(basX_trans(nonNAidx(nonNAidx < length(x0)),:),[r_wd '\X.csv'])
writematrix(ones(length(nonNAidx(nonNAidx < length(x0))), 1),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = diag([ones(length(theta0) - 1,1)*1e-4;1e-8]);
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
%

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

% plot
meanRange = [min([CMP_mean(:); CMP_mean_fit_trans(:)])...
    max([CMP_mean(:); CMP_mean_fit_trans(:)])];


% W_fit_all = -inv(hess_tmp);
% W_fit = zeros(2,2,T);
% for k = 1:T
%    W_fit(:,:,k) = W_fit_all((2*(k-1)+1):(2*k), (2*(k-1)+1):(2*k));
% end
[~,W_fit] =...
    ppasmoo_compoisson_fisher_na(theta01, spk_vec', basX_trans, ones(Tall, 1),...
    W01,eye(length(theta01)),Qoptmatrix);
[var_rate_exact, var_rate_app] = varParam(basX_trans, ones(Tall, 1), theta_fit, W_fit);

CMP_var_fit_trans = reshape(var_rate_exact, [], kStep);

%% let's do Poisson
b0 = glmfit(basX_trans(nonNAidx(nonNAidx < length(x0)),:),...
    spk_vec(nonNAidx(nonNAidx < length(x0)))','poisson','constant','off');
[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp_nan(spk_vec,basX_trans, b0,...
eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(basX_trans, 2)));
f = @(Q) helper_poisson_nan(Q, theta02, spk_vec,...
    basX_trans, W02, eye(length(theta02)));
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(basX_trans, 2))),...
    QUB*ones(1, min(2, size(basX_trans, 2))), [], options);

Qoptmatrix2 = diag([Qopt2(1) Qopt2(2)*ones(1, size(basX_trans, 2)-1)]);

gradHess_tmp = @(vecTheta) gradHessTheta_Poisson_nan(vecTheta,...
    basX_trans, theta02, W02,...
    eye(length(theta02)), Qoptmatrix2, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], Tall);

lam_poi = exp(sum(basX_trans .* theta_fit2', 2));
[~,W_fit_poi] =...
    ppasmoo_poissexp_nan(spk_vec,basX_trans, theta02,...
    W02,eye(length(theta02)),Qoptmatrix2);
lamVar_poi = zeros(Tall, 1);
for kk = 1:Tall
    VarTmp = basX_trans(kk,:)*W_fit_poi(:,:,kk)*basX_trans(kk,:)';
    ETmp = basX_trans(kk,:)*theta_fit2(:,kk);
    lamVar_poi(kk) = (exp(VarTmp)-1)*exp(2*ETmp+VarTmp);
end

POI_mean_fit_trans = reshape(lam_poi, [], kStep);
POI_var_fit_trans = reshape(lamVar_poi,[],kStep);

%% MSE
true_mean = CMP_mean;

% mean of (\hat{mu} - mu)^2
mean((CMP_mean_fit_trans - true_mean).^2, 'all')
mean((POI_mean_fit_trans - true_mean).^2, 'all')

% figure(1)
% subplot(1,3,1)
% imagesc(CMP_mean_fit_trans - true_mean)
% cLim = caxis;
% colorbar()
% title('CMP - true')
% subplot(1,3,2)
% imagesc(CMP_mean_fit_trans + sqrt(CMP_var_fit_trans) - true_mean)
% colorbar()
% title('(CMP + 1sd) - true')
% set(gca,'CLim',cLim)
% subplot(1,3,3)
% imagesc(CMP_mean_fit_trans - sqrt(CMP_var_fit_trans) - true_mean)
% colorbar()
% title('(CMP - 1sd) - true')
% set(gca,'CLim',cLim)
% 
% 
% figure(2)
% subplot(1,3,1)
% imagesc(POI_mean_fit_trans - true_mean)
% set(gca,'CLim',cLim)
% colorbar()
% title('Pisson - true')
% subplot(1,3,2)
% imagesc(POI_mean_fit_trans + sqrt(POI_var_fit_trans) - true_mean)
% colorbar()
% title('(Pisson + 1sd) - true')
% set(gca,'CLim',cLim)
% subplot(1,3,3)
% imagesc(POI_mean_fit_trans - sqrt(POI_var_fit_trans) - true_mean)
% colorbar()
% title('(Pisson - 1sd) - true')
% set(gca,'CLim',cLim)


% see spk
% training
nanmean((CMP_mean_fit_trans - spk_train).^2, 'all')
nanmean((POI_mean_fit_trans - spk_train).^2, 'all')

% test
nanmean((CMP_mean_fit_trans - spk_test).^2, 'all')
nanmean((POI_mean_fit_trans - spk_test).^2, 'all')

%% plot
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure3';
cd(plotFolder)

% FR
FR_true = figure;
imagesc(CMP_mean)
colorbar
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'CLim',meanRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_true,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(FR_true, '1_FR_true.svg')
saveas(FR_true, '1_FR_true.png')


FR_cmp = figure;
imagesc(CMP_mean_fit_trans)
colorbar
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'CLim',meanRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_cmp,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(FR_cmp, '2_FR_cmp.svg')
saveas(FR_cmp, '2_FR_cmp.png')

FR_poi = figure;
imagesc(POI_mean_fit_trans)
colorbar
xlabel('Trial')
ylabel('Orientation (degree)')
set(gca,'CLim',meanRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_poi,'PaperUnits','inches','PaperPosition',[0 0 3 3])
saveas(FR_poi, '3_FR_poi.svg')
saveas(FR_poi, '3_FR_poi.png')

% max position
posLen = length(x0);
maxSpk = zeros(kStep,1);
maxPos = zeros(kStep,1);
for t = 1:kStep
    idxTmp = ((t-1)*posLen + 1): (t*posLen);
    [maxSpk(t), ~] = max(CMP_mean(:, t));
    maxPos(t) = find(CMP_mean == maxSpk(t));
end

maxFR = figure;
hold on
l1 = plot(CMP_mean(maxPos));
l2 = plot(CMP_mean_fit_trans(maxPos), 'r');
plot(CMP_mean_fit_trans(maxPos)+ sqrt(CMP_var_fit_trans(maxPos)), 'r--', 'LineWidth', 1)
plot(CMP_mean_fit_trans(maxPos)- sqrt(CMP_var_fit_trans(maxPos)), 'r--', 'LineWidth', 1)

l3 = plot(POI_mean_fit_trans(maxPos), 'b');
plot(POI_mean_fit_trans(maxPos)+ sqrt(POI_var_fit_trans(maxPos)), 'b--', 'LineWidth', 1)
plot(POI_mean_fit_trans(maxPos)- sqrt(POI_var_fit_trans(maxPos)), 'b--', 'LineWidth', 1)
hold off
xlabel('Trial')
ylabel('Firing Rate')
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(maxFR,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(maxFR, '4_maxFR.svg')
saveas(maxFR, '4_maxFR.png')






