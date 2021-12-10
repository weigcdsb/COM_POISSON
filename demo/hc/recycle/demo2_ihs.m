addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
%
%%
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
neuron = 54;

%% EDA
pos_raw = (position_circular-1)*cam2cm;
spk_raw = spike_counts(:,neuron);

% no coarse bin
% pos = pos_raw;
% spk = spk_raw;

% coarse bin
bin = 10;
pos = zeros(ceil(length(pos_raw)/bin), 1);
spk = zeros(ceil(length(pos_raw)/bin), 1);
for k = 1:ceil(length(pos_raw)/bin)
    raw_idx = (bin*(k-1) + 1):min((bin*k), length(pos_raw));
    pos(k) = mean(pos_raw(raw_idx));
    spk(k) = sum(spk_raw(raw_idx));
end

smoothing = 10;
mean_smoo = zeros(1, length(spk) - smoothing + 1);
var_smoo = zeros(1, length(spk) - smoothing + 1);
for k = 1:(length(spk) - smoothing + 1)
    idx = k:(k + smoothing - 1);
    mean_smoo(k) = mean(spk(idx));
    var_smoo(k) = var(spk(idx));
end

t = linspace(0,size(position,1),size(position,1))/5/60;
t = t+mean(diff(t))/2;

subplot(1, 3, 1)
% plot(t,position_realigned(:,1)*cam2cm,'k')
% hold on
% plot(t(spike_counts(:,neuron)>0),position_realigned(spike_counts(:,neuron)>0,1)*cam2cm,'r.')
% hold off
plot(spk)
title('spikes')

subplot(1, 3, 2)
hold on
% plot(spk, 'Color', [1, 0.5, 0, 0.2])
plot(mean_smoo, 'b', 'LineWidth', 1)
title('window-mean')
hold off

subplot(1, 3, 3)
hold on
plot(var_smoo./mean_smoo, 'b', 'LineWidth', 1)
title('window-fano factor')
hold off

%%

nknots=32;
Gnknots=1;

% X = getCubicBSplineBasis((pos - min(pos))/range(pos),nknots,false);
% G = getCubicBSplineBasis((pos - min(pos))/range(pos),Gnknots,false);

X = getCubicBSplineBasis((pos + 250)/500*2*pi,nknots,false);
G = getCubicBSplineBasis((pos + 250)/500*2*pi,Gnknots,false);

%% sanity check the place field to make sure nknots is reasonable...

vispos = linspace(0,2*pi,256);
visX = getCubicBSplineBasis(vispos,nknots,true);
figure(3)

clf
subplot(3,2,1:2)
plot(t,(position_circular(:,1)-1)*cam2cm,'k')
hold on
plot(t(spike_counts(:,neuron)>0),(position_circular(spike_counts(:,neuron)>0,1)-1)*cam2cm,'r.')
box off
hold off
axis tight

subplot(3,2,3)
plot(pos,spk,'.')
b = glmfit(X,spk,'poisson','constant','off');
hold on
plot(vispos'/2/pi*500-250,exp(visX*b))
hold off
axis tight

subplot(3,2,5)
plot(vispos'/2/pi*500-250,visX)
axis tight

%%
nCMP = 100;
writematrix(spk(1:nCMP), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\y.csv')
writematrix(X(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\X.csv')
writematrix(G(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\G.csv')

%% fit
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmpreg.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmp_t1.csv');

windType = 'forward';

Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots + 1,1)]); % multi G

% fit smoother twice
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk', X, G,...
    eye(length(theta0)),eye(length(theta0)),Q, 5, windType); % initial: use window 5?

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-4*ones(1, min(2, size(X, 2))+ min(2, size(G, 2)))];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2(Q, theta02, spk',X,G,...
    W02,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt(1) Qopt(2)*ones(1, size(X, 2)-1)];
Q_nu = [Qopt(3) Qopt(4)*ones(1, size(G, 2) - 1)];
Qoptmatrix = diag([Q_lam Q_nu]);

[theta_fit,W_fit] =...
    ppasmoo_compoisson_v2_window_fisher(theta02, spk', X, G,...
    W02,eye(length(theta0)),Qoptmatrix, 50, windType);

% [theta_fit,W_fit] =...
%     ppasmoo_compoisson_v2_window_fisher(theta02, spk', X, G,...
%     W02,eye(length(theta0)),Q, 20, windType);

save('C:\Users\gaw19004\Desktop\COM_POI_data\hc_54_QTune_bin.mat')

subplot(1,2,1)
plot(theta_fit(1:(nknots+1), :)')
title('beta')
subplot(1,2,2)
plot(theta_fit((nknots+2):end, :)')
title('gamma')


%%
lam = exp(sum(X .* theta_fit(1:(nknots+1), :)',2));
nu = exp(sum(G .* theta_fit((nknots+2):end, :)',2));

CMP_mean = zeros(size(lam, 1), 1);
CMP_var = zeros(size(lam, 1), 1);

for k = 1:length(lam)
    logcum_app = logsum_calc(lam(k), nu(k), 500);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    CMP_mean(k) = exp(log_A - log_Z);
    CMP_var(k) = exp(log_B - log_Z) - CMP_mean(k)^2;
end


subplot(1, 2, 1)
hold on
plot(spk, 'b', 'LineWidth', 1)
% plot(mean_smoo, 'b', 'LineWidth', 1)
plot(CMP_mean, 'r', 'LineWidth', 1)
title('obs-mean')
hold off

subplot(1, 2, 2)
hold on
plot(var_smoo./mean_smoo, 'b', 'LineWidth', 1)
plot(CMP_var./CMP_mean, 'r', 'LineWidth', 1)
title('fano factor')
hold off

