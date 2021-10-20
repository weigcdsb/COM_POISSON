addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

%%
clear all;close all;clc
rng(4)
k = 200;
X_lam = ones(k, 1);
G_nu = ones(k, 1);

theta_true = zeros(2,k);

% constant
theta_true(1,:) = 1.5*ones(1,k); % min = 0.5, max = 3
theta_true(2,:) = 2*ones(1,k); % min = -.5, max = 2.5


lamSeq = exp(X_lam'.*theta_true(1,:));
nuSeq = exp(G_nu'.*theta_true(2,:));

spk_vec = com_rnd(lamSeq, nuSeq);
[mean_true,var_true] = getMeanVar(lamSeq, nuSeq);

subplot(1,3,1)
plot(lamSeq)
title('\lambda')
subplot(1,3,2)
plot(nuSeq)
title('\nu')
subplot(1,3,3)
hold on
plot(spk_vec)
plot(mean_true, 'LineWidth', 2)
hold off
title('spks')

%% let's do constant nu
iterMax = 1000;
nu_trace = ones(iterMax, 1);
theta_lam = ones(1,k, iterMax);
theta_lam(:,:,1) = log(mean(spk_vec(1:20)))*theta_lam(:,:,1);


for g = 2:iterMax
   
    % (1) update lambda_i
    theta_tmp = theta_lam(:,:,g-1);
    gradHess_tmp = @(vecTheta) gradHessTheta_CMP_fixNu(vecTheta, X_lam,...
        nu_trace(g-1),theta_lam(:,1,g-1),1e-2,1, 1e-3, spk_vec);
    [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
    if(sum(isnan(theta_newton_vec)) ~= 0)
        disp('use smoother')
        [theta_tmp,~,~] = ppasmoo_cmp_fixNu(spk_vec,X_lam,nu_trace(g-1),...
            theta_lam(:,1,g-1),1e-2,1,1e-3);
        [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
    end
    theta_lam(:,:,g)  = reshape(theta_newton_vec, [], k);
    
    % (2) update nu
    gradHess_tmp = @(nu) gradHessNu(nu, X_lam, theta_lam(:,:,g), spk_vec);
    nu_trace(g) = newtonGH(gradHess_tmp, nu_trace(g-1), 1e-10, 1000);
    
    disp(norm(nu_trace(g) - nu_trace(g-1)))
   
    if(norm(nu_trace(g) - nu_trace(g-1)) < 1e-6)
        break;
    end
end


plot(exp(theta_lam(:,:,g)'))
nu_trace(g)











