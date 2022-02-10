addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];
%%
rng(1)
T = 2000;
n = 1; % number of independent observations
t = linspace(0,1,T);

X_lam = ones(T, 1);
G_nu = ones(T, 1);

theta_true = zeros(T,2);

target_mean = 4;
theta_true(:,2) = 10*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2) - 1; % 8 .4
% plot(theta_true(:,2))
nu_true = exp(G_nu.*theta_true(:, 2));
theta_true(:,1) = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true));
Q=diag([1e-3 1e-3]);


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));
spk_vec = com_rnd(lam_true, nu_true);
theo_mean = zeros(1,T);
theo_var = zeros(1,T);

for k = 1:T
    [theo_mean(k), theo_var(k), ~, ~, ~, ~] = ...
        CMPmoment(lam_true(k), nu_true(k), 1000);
end

hold on
plot(spk_vec)
plot(theo_mean)
hold off

plot(theo_var)

plot(theo_var./theo_mean)
close all;

%%
theta0 = theta_true(1,:)';
Q = 1e-4*eye(length(theta0));
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec, X_lam, G_nu,...
    eye(length(theta0)),eye(length(theta0)),Q);

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(X_lam, 2))+ min(2, size(G_nu, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(Q, theta01, spk_vec, X_lam, G_nu,...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);
Qoptmatrix = diag(Qopt);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, X_lam, G_nu,...
    theta01, W01,eye(length(theta01)), Qoptmatrix, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit = reshape(theta_newton_vec, [], T);

lam_fit = zeros(1, T);
nu_fit = zeros(1, T);
CMP_mean_fit = zeros(1, T);
CMP_var_fit = zeros(1, T);
logZ_fit = zeros(1,T);

for m = 1:T
    lam_fit(m) = exp(X_lam(m,:)*theta_fit(1, m));
    nu_fit(m) = exp(G_nu(m,:)*theta_fit(2, m));
    [CMP_mean_fit(m), CMP_var_fit(m), ~, ~, ~, logZ_fit(m)] = ...
            CMPmoment(lam_fit(m), nu_fit(m), 1000);
end

figure(1)
hold on
plot(theo_mean)
plot(CMP_mean_fit)
hold off

figure(2)
hold on
plot(theo_var./theo_mean, 'LineWidth', 2)
plot(CMP_var_fit./CMP_mean_fit, 'LineWidth', 2)
hold off


% variance of mean
W_fit_all = -inv(hess_tmp);
W_fit = zeros(2,2,T);
for k = 1:T
   W_fit(:,:,k) = W_fit_all((2*(k-1)+1):(2*k), (2*(k-1)+1):(2*k));
end
[var_rate_exact, var_rate_app] = varParam(X_lam, G_nu, theta_fit, W_fit);


%% what about Poisson?
b0 = glmfit(X_lam,spk_vec','poisson','constant','off');
[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp_nan(spk_vec,X_lam, b0,eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(X_lam, 2)));
f = @(Q) helper_poisson_nan(Q, theta02, spk_vec,...
    X_lam, W02, eye(length(theta02)));
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(X_lam, 2))),QUB*ones(1, min(2, size(X_lam, 2))), [], options);

% Qopt2 = Qoptmatrix(1,1);
gradHess_tmp = @(vecTheta) gradHessTheta_Poisson_nan(vecTheta, X_lam, theta02, W02,...
    eye(length(theta02)), Qopt2, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], T);
% plot(theta_fit2)

lam_poi = exp(sum(X_lam .* theta_fit2', 2));
W_fit_poi = diag(-inv(hess_tmp));
lamVar_poi = (exp(W_fit_poi) - ones(T,1)).*exp(2*theta_fit2' + W_fit_poi);


figure(1)
hold on
plot(spk_vec, 'Color', [0.4, 0.4, 0.4, 0.2])
plot(theo_mean, 'k', 'LineWidth', 2)
plot(CMP_mean_fit, 'b', 'LineWidth', 2)
plot(lam_poi, 'r', 'LineWidth', 2)
hold off

figure(2)
hold on
plot(lamVar_poi)
plot(var_rate_exact)
hold off
close all;

%% let's plot
% plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure2';
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure2\Poi_Q_tune';
cd(plotFolder)

FitY_CMP = figure;
hold on
plot(spk_vec, 'Color', [0.4, 0.4, 0.4, 0.2])
plot(theo_mean, 'k', 'LineWidth', 2)
plot(CMP_mean_fit, 'r', 'LineWidth', 2)
plot(CMP_mean_fit+ sqrt(CMP_var_fit), 'r--', 'LineWidth', 1)
plot(CMP_mean_fit- sqrt(CMP_var_fit), 'r--', 'LineWidth', 1)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FitY_CMP,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FitY_CMP, '1_FitY_CMP.svg')
saveas(FitY_CMP, '1_FitY_CMP.png')



FitY_Poi = figure;
hold on
plot(spk_vec, 'Color', [0.4, 0.4, 0.4, 0.2])
plot(theo_mean, 'k', 'LineWidth', 2)
plot(lam_poi, 'r', 'LineWidth', 2)
plot(lam_poi+ sqrt(lam_poi), 'r--', 'LineWidth', 1)
plot(lam_poi- sqrt(lam_poi), 'r--', 'LineWidth', 1)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FitY_Poi,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FitY_Poi, '2_FitY_Poi.svg')
saveas(FitY_Poi, '2_FitY_Poi.png')

FitY_all = figure;
hold on
plot(spk_vec, 'Color', [0.4, 0.4, 0.4, 0.2])
plot(theo_mean, 'k', 'LineWidth', 2)
plot(CMP_mean_fit, 'r', 'LineWidth', 2)
plot(CMP_mean_fit+ sqrt(CMP_var_fit), 'r--', 'LineWidth', 1)
plot(CMP_mean_fit- sqrt(CMP_var_fit), 'r--', 'LineWidth', 1)
plot(lam_poi, 'b', 'LineWidth', 2)
plot(lam_poi+ sqrt(lam_poi), 'b--', 'LineWidth', 1)
plot(lam_poi- sqrt(lam_poi), 'b--', 'LineWidth', 1)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FitY_all,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FitY_all, '3_FitY_all.svg')
saveas(FitY_all, '3_FitY_all.png')

FF = figure;
hold on
plot(theo_var./theo_mean, 'LineWidth', 2)
plot(CMP_var_fit./CMP_mean_fit, 'LineWidth', 2)
yline(1, 'LineWidth', 2);
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FF,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FF, '4_FF.svg')
saveas(FF, '4_FF.png')


meanComp = figure;
hold on
plot(theo_mean, 'k', 'LineWidth', 2)
plot(CMP_mean_fit, 'r', 'LineWidth', 2)
plot(CMP_mean_fit+ sqrt(var_rate_exact'), 'r--', 'LineWidth', 1)
plot(CMP_mean_fit- sqrt(var_rate_exact'), 'r--', 'LineWidth', 1)
plot(lam_poi, 'b', 'LineWidth', 2)
plot(lam_poi+ sqrt(lamVar_poi), 'b--', 'LineWidth', 1)
plot(lam_poi- sqrt(lamVar_poi), 'b--', 'LineWidth', 1)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(meanComp,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(meanComp, '5_mean.svg')
saveas(meanComp, '5_mean.png')


VarComp = figure;
hold on
plot(var_rate_exact, 'r', 'LineWidth', 2)
plot(lamVar_poi, 'b', 'LineWidth', 2)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(VarComp,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(VarComp, '6_var.svg')
saveas(VarComp, '6_var.png')


