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
theta_true(:,2) = 8*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2) - 0.4;
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

%% let's plot
plotFolder = 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure2';
cd(plotFolder)

MFR = figure;
hold on
plot(spk_vec, 'Color', [0.4, 0.4, 0.4, 0.2])
plot(theo_mean, 'b', 'LineWidth', 2)
plot(CMP_mean_fit, 'r', 'LineWidth', 2)
% plot(CMP_mean_fit+ 1.96*sqrt(var_rate_app'), 'r--', 'LineWidth', 2)
% plot(CMP_mean_fit- 1.96*sqrt(var_rate_app'), 'r--', 'LineWidth', 2)
plot(CMP_mean_fit+ sqrt(var_rate_exact'), 'r--', 'LineWidth', 2)
plot(CMP_mean_fit- sqrt(var_rate_exact'), 'r--', 'LineWidth', 2)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(MFR,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(MFR, '1_MFR.svg')
saveas(MFR, '1_MFR.png')


FF = figure;
hold on
plot(theo_var./theo_mean, 'LineWidth', 2)
plot(CMP_var_fit./CMP_mean_fit, 'LineWidth', 2)
hold off
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off

set(FF,'PaperUnits','inches','PaperPosition',[0 0 5 3])
saveas(FF, '2_FF.svg')
saveas(FF, '2_FF.png')

