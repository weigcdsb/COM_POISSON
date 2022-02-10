addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%% true underlying mean & FF
rng(1)
nknots = 5;
x0 = linspace(0,1,100);
basX = getCubicBSplineBasis(x0,nknots,false);

T = 100;
dt = 1;
kStep = T/dt;

bits = sign(mod(1:24,2)-0.5);
nu_fluc = ones(1,kStep)*Inf;
while(sum(abs(nu_fluc) > 1.5) > 0)
    nu_fluc = interp1(linspace(0,1,24),randsample(bits,24)+randn(1,24),linspace(0,1,T),'spline');
end

gam = repmat(nu_fluc'/2, 1, kStep)';
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

beta(1,:) = 2;
beta(4,:) = 5; %10;

% lam = exp(nu.*log(exp(basX*beta + (nu - 1)./(2.*nu))));
% imagesc(basX*beta)
% colorbar()

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
rng(2)

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        spk(m,n) = com_rnd(lam(m,n), nu(m,n), 1);
        
    end
end


subplot(2,2,1)
imagesc(spk)
colorbar()
title('spk')
subplot(2,2,2)
imagesc(CMP_var./CMP_mean)
colorbar()
title('FF')
subplot(2,2,3)
plot(lam(:))
title('lam')
subplot(2,2,4)
plot(nu(:))
title('nu')

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

% QLB = -20;
% QUB = 1;
% Q0 = -6*ones(1, min(2, size(basX_trans, 2))+ min(2, size(ones(Tall, 1), 2)));
% Q0 = [-10 -5 -5];
% DiffMinChange = 0.01;
% DiffMaxChange = 1;
% MaxFunEvals = 500;
% MaxIter = 500;
% 
% f = @(Q) helper_na(exp(Q), theta01, spk_vec',basX_trans,ones(Tall, 1),...
%     W01,eye(length(theta0)));
% options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
%     'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
% Qopt = fmincon(f,Q0,[],[],[],[],...
%     QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);
% Qopt = exp(Qopt);
% Qopt = [10e-6 2*10e-4 2*10e-4];


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
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure2';
cd(plotFolder)

meanRange = [min([CMP_mean(:); CMP_mean_fit_trans(:)])...
    max([CMP_mean(:); CMP_mean_fit_trans(:)])];
ffRange = [min([CMP_var(:)./CMP_mean(:); CMP_ff_fit_trans(:)])...
max([CMP_var(:)./CMP_mean(:); CMP_ff_fit_trans(:)])];

% subplot(2,2,1)
FR_true = figure;
imagesc(1:dt:T, angle, CMP_mean)
% title('Mean Firing Rate')
xlabel('Trial')
ylabel('Orientation (degree)')
% colormap(flipud(gray(256)));
colorbar;
set(gca,'CLim',meanRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_true,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FR_true, '1_FR_true.svg')
saveas(FR_true, '1_FR_true.png')

% subplot(2,2,2)
FF_true = figure;
imagesc(1:dt:T, angle, CMP_var./CMP_mean)
xlabel('Trial')
ylabel('Orientation (degree)')
% colormap(flipud(gray(256)));
colorbar;
set(gca,'CLim',ffRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FF_true,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FF_true, '2_FF_true.svg')
saveas(FF_true, '2_FF_true.png')

% subplot(2,2,3)
FR_fit = figure;
imagesc(1:dt:T, angle, CMP_mean_fit_trans)
xlabel('Trial')
ylabel('Orientation (degree)')
% colormap(flipud(gray(256)));
colorbar;
set(gca,'CLim',meanRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FR_fit,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FR_fit, '3_FR_fit.svg')
saveas(FR_fit, '3_FR_fit.png')

% subplot(2,2,4)
FF_fit = figure;
imagesc(1:dt:T, angle, CMP_ff_fit_trans)
xlabel('Trial')
ylabel('Orientation (degree)')
% colormap(flipud(gray(256)));
colorbar;
set(gca,'CLim',ffRange)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FF_fit,'PaperUnits','inches','PaperPosition',[0 0 4 3])
saveas(FF_fit, '4_FF_fit.svg')
saveas(FF_fit, '4_FF_fit.png')



