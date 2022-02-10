addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%%
rng(12)
nknots = 5;
x0 = linspace(0,1,50);
basX = getCubicBSplineBasis(x0,nknots,false);

T = 100;
dt = 1;
kStep = T/dt;

mupf = interp1(linspace(1,T,10),randn(1,10)/10+0.5,linspace(1,T,T),'spline');
[tm,xm]=meshgrid(1:T,x0);

lam = exp(-(xm-mupf).^2/2/0.05)*20;
lam(lam < 10) = 0;% max(lam(lam < 8)-3, 0);
imagesc(lam)
colorbar()

spk = poissrnd(lam);
spk(lam < 10) = spk(lam < 10) + round((rand(size(spk(lam < 10))) -0.5)*5);
spk(lam >= 10) = spk(lam >= 10) + round((rand(size(spk(lam >= 10))) - 0.5)*10);
spk(spk < 0) =0;

subplot(1,2,1)
imagesc(lam)
colorbar
subplot(1,2,2)
imagesc(spk)
colorbar

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
meanRange = [min([lam(:); CMP_mean_fit_trans(:)])...
    max([lam(:); CMP_mean_fit_trans(:)])];

%% let's do Poisson
b0 = glmfit(basX_trans,spk_vec','poisson','constant','off');
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
POI_mean_fit_trans = reshape(lam_poi, [], kStep);


% plot
subplot(3,1,1)
imagesc(lam)
colorbar
title('true')
set(gca,'CLim',meanRange)
subplot(3,1,2)
imagesc(CMP_mean_fit_trans)
colorbar
set(gca,'CLim',meanRange)
title('cmp')
subplot(3,1,3)
imagesc(POI_mean_fit_trans)
colorbar
set(gca,'CLim',meanRange)
title('Poisson')

