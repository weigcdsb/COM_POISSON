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
neuron=13; 
% 72 46 13
ry = reshape(trial_y(:,neuron),100,[]);
[~,theta_idx]=sort(theta);

%
nknots=7;
Gnknots=7;
X = getCubicBSplineBasis(theta',nknots,true);
G = getCubicBSplineBasis(theta',Gnknots,true);

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

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

%% no Q-tune
% Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-5,Gnknots,1)]); % single G
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-5,Gnknots + 1,1)]); % multi G

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2(theta0, trial_y(:,neuron)', Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),Q);

% do smoothing twice
theta021 = theta_fit_tmp(:, 1);
W021 = W_fit_tmp(:, :, 1);

[theta_fit1,W_fit1] =...
    ppasmoo_compoisson_v2(theta021, trial_y(:,neuron)', Xb, Gb,...
    W021,eye(length(theta0)),Q);

%% Q-tune version

QLB = 1e-8;
QUB = 1e-4;
Q0 = QLB*ones(1, length(theta0));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 100;
MaxIter = 25;

% way 1: search Q with W0 = I, and use the optimized Q to do double
% smoothing...
f = @(Q) helper_2d(Q, theta0, trial_y(1:6000,neuron)',Xb,Gb,...
    eye(length(theta0)),eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta0)),QUB*ones(1, length(theta0)), [], options);

% do smoothing twice
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2(theta0, trial_y(:,neuron)', Xb, Gb,...
    eye(length(theta0)),eye(length(theta0)),diag(Qopt));
theta022 = theta_fit_tmp(:, 1);
W022 = W_fit_tmp(:, :, 1);
[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v2(theta022, trial_y(:,neuron)', Xb, Gb,...
    W022,eye(length(theta0)),diag(Qopt));

% way2: use previous W0, optimize Q and do smoothing once...
f = @(Q) helper_2d(Q, theta021, trial_y(1:6000,neuron)',Xb,Gb,...
    W021,eye(length(theta021)));
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(theta021)),QUB*ones(1, length(theta021)), [], options);
[theta_fit3,W_fit3] =...
    ppasmoo_compoisson_v2(theta021, trial_y(:,neuron)', Xb, Gb,...
    W021,eye(length(theta0)),diag(Qopt));

save('v1.mat')

%% parameter tracks...
theta_fit = theta_fit3;

param = figure;
subplot(1,2,1)
plot(theta_fit(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit((nknots+2):end, :)')
title('gamma')

% saveas(param, 'param_track_noQTune.png')
% saveas(param, 'param_track_QTune_twice.png')
saveas(param, 'param_track_QTune_once.png')

%% calculate mean & var

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

heat = figure;
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

% saveas(heat, 'heat_noQTune.png')
% saveas(heat, 'heat_QTune_twice.png')
saveas(heat, 'heat_QTune_once.png')

%%
line = figure;
subplot(1,2,1)
plot(ry_hat(theta_idx,:))
% plot(ry_hat(theta_idx,10:end))
title('Mean')
subplot(1,2,2)
plot(ry_var(theta_idx,:)./ry_hat(theta_idx,:))
% plot(ry_var(theta_idx,10:end)./ry_hat(theta_idx,10:end))
title('Fano Factor')

% saveas(line, 'line_noQTune.png')
% saveas(line, 'line_QTune_twice.png')
saveas(line, 'line_QTune_once.png')
