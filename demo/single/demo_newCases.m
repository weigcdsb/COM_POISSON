% addpath(genpath('D:\GitHub\COM_POISSON'));
addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%%
rng(123)
T = 60; %when T = 10, not enough samples, not robust
dt = 0.005; % bin length (s)
n = 1; % number of independent observations
t = linspace(0,1,T/dt);

X_lam = ones(T/dt, 1);
G_nu = ones(T/dt, 1);

theta_true = zeros(T/dt,2);

% % Case 1 -- Mean increase - poisson model (good)
theta_true(:,1) = (t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1;
theta_true(:,2) = 0;
Q=diag([1e-2 1e-6]);

% % Case 2 -- Var decrease - constant(ish) mean (not bad)
% target_mean = 10; 
% theta_true(:,2) = 5*(t-0.2)/.05.*exp(-(t-0.2)/.05).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% % theta_true(:,1) = log(10.^nu_true); % better approximation...
% theta_true(:,1) = nu_true.*log(target_mean + (nu_true - 1)./ (2*nu_true));
% Q=diag([1e-3 1e-3]);

% Case 3 -- Mean increase + Var decrease
% % seems mean increase --> var increase also, confused
% theta_true(:,2) = 3*(t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2);
% nu_true = exp(G_nu.*theta_true(:, 2));
% % theta_true(:,1) = log(matchMean(exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1),nu_true));
% % to run fast... use approximation again
% target_mean = exp((t-0.2)/.1.*exp(-(t-0.2)/.1).*(t>.2)*6+1);
% % hold on
% % plot(nu_true.*log(target_mean' + (nu_true - 1)./ (2*nu_true)))
% % plot(theta_true(:,1))
% % hold off
% theta_true(:,1) = nu_true.*log(target_mean' + (nu_true - 1)./ (2*nu_true));
% Q=diag([1e-3 1e-3]);

lam_true = exp(X_lam.*theta_true(:, 1));
nu_true = exp(G_nu.*theta_true(:, 2));

% subplot(2, 2, 1)
% plot(theta_true(:,1))
% subplot(2, 2, 2)
% plot(theta_true(:,2))
% subplot(2, 2, 3)
% plot(lam_true)
% subplot(2, 2, 4)
% plot(nu_true)

spk_vec = com_rnd(lam_true, nu_true);
[theo_mean,theo_var]=getMeanVar(lam_true,nu_true);

% subplot(1, 2, 1)
% plot(theo_mean)
% subplot(1, 2, 2)
% plot(theo_var)

figure(3)
subplot(2,3,1)
plot(mean(spk_vec, 1))
hold on
plot(theo_mean, 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')
sgtitle('case1') 



% fit
% initialize...
% max_init = 100;
% theta0 = [log(mean(spk_vec(1:max_init))); 0];
% W0 = diag([1 1]);
% tf_path=[];
% 
% F = diag([1 1]);
% 
% for rep=1:10
%     ridx = randperm(max_init);
% %     [theta_fit1,W_fit1] =...
% %         ppafilt_compoisson_v2(theta0, spk_vec(ridx),X_lam,G_nu,...
% %         W0,F,diag([1e-3 1e-3])); % why filtering, not smoothing?
%     [theta_fit1,W_fit1] =...
%         ppasmoo_compoisson_v2(theta0, spk_vec(ridx),X_lam,G_nu,...
%         W0,F,diag([1e-3 1e-3]));
%     theta0 = theta_fit1(:,end);
%     W0 = W_fit1(:,:,end);
%     tf_path(:,rep)=theta0;
% end
% 
% plot(tf_path')
% theta0 = mean(tf_path, 2);

% way 2: run smoothing twice for the first 100 spikes?
max_init = 100;
spk_tmp = spk_vec(1:max_init);
theta0_tmp = [log(mean(spk_tmp)); 0];
W0_tmp = diag([1 1]);
F = diag([1 1]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2(theta0_tmp, spk_tmp,X_lam,G_nu,...
    W0_tmp,F,diag([1e-3 1e-3]));
theta0 = theta_fit_tmp(:, 1);
W0 = W_fit_tmp(:, :, 1);

%
% figure(2)
% nv=0:max(spk_vec);
% histogram(spk_vec(1:max_init),nv-.5,'EdgeColor','none','Normalization','pdf')
% hold on
% plot(nv,com_pdf(nv,exp(theta0(1)),exp(theta0(2))));
% plot(nv,poisspdf(nv,mean(spk_vec(1:max_init))));
% hold off

%
QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, length(theta0));
DiffMinChange = QLB;
DiffMaxChange = QUB;
MaxFunEvals = 200;
MaxIter = 25;

f = @(Q) helper_2d(Q, theta0, spk_vec,X_lam,G_nu,...
    W0,F);

% fmincon
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);

% % check if unimodal?
% nQ = 10;
% Qvec = logspace(log10(QLB), log10(QUB), nQ);
% llhdmesh = zeros(nQ, nQ);
% for j = 1:nQ
%     for k = 1:nQ
%         fprintf('Qbeta0 %02i/%02i... Qwtlong %02i/%02i...', j, nQ, k, nQ)
%         [theta_fit,W_fit, lam_pred, nu_pred, log_Z_pred] =...
%             ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
%             W0,F,diag([Qvec(j) Qvec(k)]));
% %         [theta_fit,W_fit, lam_pred, nu_pred, log_Z_pred] =...
% %             ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
% %             W0,F,diag([Qvec(j) Qvec(k)]));
%         if(length(log_Z_pred) == size(spk_vec, 2))
%             llhd_pred = sum(spk_vec.*log((lam_pred+(lam_pred==0))) -...
%                 nu_pred.*gammaln(spk_vec + 1) - log_Z_pred);
%             fprintf('llhd %.02f... \n', llhd_pred)
%             llhdmesh(j ,k) = llhd_pred;
%         else
%            llhdmesh(j ,k) = -Inf; 
%         end
%     end
% end
% 
% % llhdmesh(find(llhdmesh < -7000)) = -7000
% 
% [qlam_indx, qnu_indx] = find(llhdmesh == max(max(llhdmesh)));
% hold on
% imagesc(log10(Qvec), log10(Qvec), llhdmesh);
% colorbar
% plot(log10(Qvec(qnu_indx)), log10(Qvec(qlam_indx)), 'o', 'Color', 'r',...
%     'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)
% plot(log10(Qopt(2)), log10(Qopt(1)), 'o', 'Color', 'b',...
%     'LineWidth', 2, 'markerfacecolor', 'b', 'MarkerSize',5)
% hold off


[theta_fit1,W_fit1] =...
    ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
    W0,F,diag(Qopt));

% [theta_fit1,W_fit1] =...
%     ppasmoo_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,Q);
%
% [theta_fit1,W_fit1] =...
%     ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,diag(Qopt));
% 
% [theta_fit1,W_fit1] =...
%     ppafilt_compoisson_v2(theta0, spk_vec,X_lam,G_nu,...
%     W0,F,Q);

[est_mean,est_var]=getMeanVar(exp(theta_fit1(1,:)),exp(theta_fit1(2,:)));

figure(3)
subplot(2,3,1)
plot(mean(spk_vec, 1))
hold on
plot(theo_mean, 'LineWidth', 2)
hold off
box off; set(gca,'TickDir','out')
ylabel('Observations')
sgtitle('case1') 

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

% save('C:\Users\gaw19004\Desktop\COM_POI_data\sim_case1.mat')
% saveas(figure(3), 'case1.png')
