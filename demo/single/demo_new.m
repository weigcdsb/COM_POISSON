addpath(genpath('D:\GitHub\COM_POISSON'));
% addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(123)
T = 10;
dt = 0.005; % bin length (s)
n = 1; % number of independent observations
t = linspace(0,1,T/dt);

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = zeros(T/dt,2);

% % Case 1 -- Mean increase - poisson model
% theta_true(:,1) = (t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1;
% theta_true(:,2) = 0;
% Q=diag([1e-2 1e-6]);

% % Case 2 -- Var decrease - constant(ish) mean
theta_true(:,2) = 5*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2);
nu_true = exp(G_nu.*theta_true(:, 2));
theta_true(:,1) = log(10.^nu_true);
Q=diag([1e-4 1e-4]);

% Case 3 -- Mean increase + Var decrease
% theta_true(:,2) = 3*(t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% theta_true(:,1) = log(matchMean(exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1),nu_true));
% Q=diag([1e-3 1e-3]);


lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));


spk_vec = com_rnd(lam_true, nu_true);
[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);

figure(3)
subplot(2,3,1)
plot(mean(spk_vec, 1))
hold on
plot(theo_mean, 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')


% fit

% initialize...
max_init = 100;
theta0 = [log(mean(spk_vec(1:max_init))); 0];
W0 = diag([1 1]);
tf_path=[];

F = diag([1 1]);

for rep=1:100
    ridx = randperm(max_init);
%     [theta_fit1,W_fit1] =...
%         ppafilt_compoisson_v2(theta0, spk_vec(ridx),X_lam,G_nu,...
%         W0,F,diag([1e-3 1e-3])); % why filtering, not smoothing?
%     theta0 = theta_fit1(:,end);
%     W0 = W_fit1(:,:,end);
    [theta_fit1,W_fit1] =...
        ppasmoo_compoisson_v2(theta0, spk_vec(ridx),X_lam,G_nu,...
        W0,F,diag([1e-3 1e-3]));
    theta0 = theta_fit1(:,1);
    W0 = W_fit1(:,:,1);
    tf_path(:,rep)=theta0;
end

plot(tf_path')
% theta0 = mean(tf_path, 2);

%
figure(2)
nv=0:max(spk_vec);
histogram(spk_vec(1:max_init),nv-.5,'EdgeColor','none','Normalization','pdf')
hold on
plot(nv,com_pdf(nv,exp(theta0(1)),exp(theta0(2))));
plot(nv,poisspdf(nv,mean(spk_vec(1:max_init))));
hold off

%
QLB = 1e-8;
QUB = 1e-3;
Q0 = QLB*ones(1, length(theta0));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 100;
MaxIter = 25;

f = @(Q) helper_2d(Q, theta0, spk_vec,X_lam,G_nu,...
    W0,F);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);


[theta_fit1,W_fit1] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt));

% [theta_fit1,W_fit1] =...
%     ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,diag(Qopt));

[est_mean,est_var]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

figure(3)
subplot(2,3,2)
plot(est_mean)
hold on
plot(theo_mean, 'LineWidth', 2)
hold off
ylabel('Mean')

subplot(2,3,5)
plot(est_var)
hold on
plot(theo_var, 'LineWidth', 2)
hold off
ylabel('Var')


subplot(2,3,4)
plot(est_var./est_mean)
hold on
plot(theo_var./theo_mean, 'LineWidth', 2)
hold off
ylim([0 4])
ylabel('Fano Factor')

subplot(2,3,3)
plot((theta_fit1(1,:)))
hold on
plot((theta_true(:,1)), 'LineWidth', 2)
hold off
ylabel('beta')

subplot(2,3,6)
plot((theta_fit1(2,:)))
hold on
plot((theta_true(:,2)), 'LineWidth', 2)
hold off
ylabel('gamma')

