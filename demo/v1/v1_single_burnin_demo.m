addpath(genpath('D:\GitHub\COM_POISSON'));
%
%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

%%
trial_x=repmat(theta',size(data.EVENTS,2),1);

trial_y=[];
c=1;

stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
        end
        t=t+stim_length;
        c=c+1;
    end
end

%%

% no augment:
repTrial = 0;
trial_y_aug = trial_y;
trial_x_aug = trial_x;

% data augment: replicate the first 5 trials
% repTrial = 10;

% way 1: (trial_1,…trial_1) + (trial_1, trial_2,…, trial_120)
% trial_y_aug = [repmat(trial_y(1:100, :), repTrial, 1); trial_y];
% trial_x_aug = [repmat(trial_x(1:100, :), repTrial, 1); trial_x];

% way 2: (trial_1,…trial_n) + (trial_1, trial_2,…, trial_120)
% trial_y_aug = [trial_y(1:(100*repTrial), :); trial_y];
% trial_x_aug = [trial_x(1:(100*repTrial), :); trial_x];

neuron=3; 
% 72 46 13
ry = reshape(trial_y(:,neuron),100,[]);
ry_aug = reshape(trial_y_aug(:,neuron),100,[]);
[~,theta_idx]=sort(theta);

%
nknots=7;
Gnknots=7;
X = getCubicBSplineBasis(theta',nknots,true);
G = getCubicBSplineBasis(theta',Gnknots,true);

Xb_aug = getCubicBSplineBasis(trial_x_aug,nknots,true);
Gb_aug = getCubicBSplineBasis(trial_x_aug,Gnknots,true);

x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

%%
trial = 1:5;
writematrix(reshape(ry(:, trial), [], 1), 'D:\GitHub\COM_POISSON\runRcode\y.csv')
writematrix(repmat(X, length(trial), 1), 'D:\GitHub\COM_POISSON\runRcode\X.csv')
writematrix(repmat(G, length(trial), 1), 'D:\GitHub\COM_POISSON\runRcode\G.csv')


%% fit the data

% cmp initials
RunRcode('D:\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'E:\software\R\R-4.0.2\bin');
theta0 = readmatrix('D:\GitHub\COM_POISSON\runRcode\cmp_t1.csv');

% poisson initials
% b = glmfit(repmat(X, length(trial), 1),reshape(ry(:, trial), [], 1),'poisson','constant','off');
% theta0 = [b; zeros(Gnknots, 1)-2]; % Poisson, single
% theta0 = [b; zeros(Gnknots + 1, 1)-2]; % Poisson, multi
%%
% Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-5,Gnknots,1)]); % single G
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-5,Gnknots + 1,1)]); % multi G

[theta_fit1,W_fit1] =...
    ppasmoo_compoisson_v2(theta0, trial_y_aug(:,neuron)', Xb_aug, Gb_aug,...
    eye(length(theta0)),eye(length(theta0)),Q);

% do smoothing twice
theta02 = theta_fit1(:, 1);
W02 = W_fit1(:, :, 1);

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta02, trial_y_aug(:,neuron)', Xb_aug, Gb_aug,...
    W02,eye(length(theta0)),Q);

theta_fit2 = theta_fit2(:, (100*repTrial + 1):end);
W_fit2 = W_fit2(:, :, (100*repTrial + 1):end);
%%
figure(2)
subplot(1,2,1)
plot(theta_fit2(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit2((nknots+2):end, :)')
title('gamma')

%% calculate mean & var

theta_fit = theta_fit2;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

lam = exp(sum(Xb .* theta_fit(1:(nknots+1), :)',2));
nu = exp(sum(Gb .* theta_fit((nknots+2):end, :)',2));

CMP_mean = zeros(size(lam, 1), size(lam, 2));
CMP_var = zeros(size(lam, 1), size(lam, 2));

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        cum_app = sum_calc(lam(m, n), nu(m, n), 500);
        Z = cum_app(1);
        A = cum_app(2);
        B = cum_app(3);
        
        CMP_mean(m, n) = A/Z;
        CMP_var(m, n) = B/Z - CMP_mean(m, n)^2;
    end
end

CMP_ff = CMP_var./CMP_mean;

%%

figure(3)
ry_hat = reshape(CMP_mean,100,[]);
ry_var = reshape(CMP_var,100,[]);
subplot(1,2,1)
imagesc(ry_hat(theta_idx,:))
% imagesc(ry_hat(theta_idx,10:end))
title('Mean')
subplot(1,2,2)
imagesc(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% imagesc(ry_var(theta_idx,10:end)./ry_hat(theta_idx,10:end))
title('Fano Factor')
colorbar

%%
figure(4)
subplot(1,2,1)
plot(ry_hat(theta_idx,:))
% plot(ry_hat(theta_idx,10:end))
title('Mean')
subplot(1,2,2)
plot(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% plot(ry_var(theta_idx,10:end)./ry_hat(theta_idx,10:end))
title('Fano Factor')


