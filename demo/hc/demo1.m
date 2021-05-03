addpath(genpath('D:\GitHub\COM_POISSON'));
%
%%
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
neuron = 27;

%%
pos = (position_circular-1)*cam2cm;
spk = spike_counts(:,neuron);

% no data augment
rep = 0;
n =1;
pos_aug = pos;
spk_aug = spk;

% data augment
% rep = 2000;
% n = 1;
% pos_aug = [repmat(pos(1:rep), n, 1); pos];
% spk_aug = [repmat(spk(1:rep), n, 1); spk];


nknots=5;
x0 = linspace(min(pos), max(pos), 500);
bas = getCubicBSplineBasis((x0-min(x0))/range(x0),nknots,false);

% figure(1)
% hold on
% plot(pos,spk,'.')
% plot(x0, bas)
% hold off

X = getCubicBSplineBasis((pos - min(pos))/range(pos),nknots,false);
[b,~,~] = glmfit(X,spk,'poisson','constant','off');

% figure(2)
% plot(pos,spk,'.')
% hold on
% plot(x0,exp(bas*b),'LineWidth',2)
% hold off

%%

nknots=4;
Gnknots=4;
X = getCubicBSplineBasis((pos - min(pos))/range(pos),nknots,false);
G = getCubicBSplineBasis((pos - min(pos))/range(pos),Gnknots,false);

X_aug = getCubicBSplineBasis((pos_aug - min(pos_aug))/range(pos_aug),nknots,false);
G_aug= getCubicBSplineBasis((pos_aug - min(pos_aug))/range(pos_aug),Gnknots,false);

nCMP = 5000;
writematrix(spk(1:nCMP), 'D:\GitHub\COM_POISSON\demo\hc\y.csv')
writematrix(X(1:nCMP, :), 'D:\GitHub\COM_POISSON\demo\hc\X.csv')
writematrix(G(1:nCMP, :), 'D:\GitHub\COM_POISSON\demo\hc\G.csv')

%% fit
RunRcode('D:\GitHub\COM_POISSON\demo\hc\cmpreg.r',...
    'E:\software\R\R-4.0.2\bin');
theta0 = readmatrix('D:\GitHub\COM_POISSON\demo\hc\cmp_t1.csv');

% Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-4,Gnknots,1)]); % single G
Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots + 1,1)]); % multi G

[theta_fit_aug,W_fit_aug] =...
    ppasmoo_compoisson_v2(theta0, spk_aug', X_aug, G_aug,...
    eye(length(theta0)),eye(length(theta0)),Q);


theta_fit = theta_fit_aug(:, (rep*n+1):end);
W_fit = W_fit_aug(:, :, (rep*n+1):end);


% subplot(1,2,1)
% plot(theta_fit_aug(1:(nknots+1), :)')
% title('beta')
% subplot(1,2,2)
% plot(theta_fit_aug((nknots+2):end, :)')
% title('gamma')

figure(3)
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

figure(4)
hold on
plot(spk)
xlabel('steps')
ylabel('Spike Counts')
plot(CMP_mean)
legend('obs.','CMP-mean')
hold off

%%
t = linspace(0,size(position,1),size(position,1))/5/60;
t = t+mean(diff(t))/2;

subplot(1,2,1)
plot(t,position_realigned(:,1)*cam2cm,'k')
hold on
plot(t(spike_counts(:,neuron)>0),position_realigned(spike_counts(:,neuron)>0,1)*cam2cm,'r.')
hold off

subplot(1,2,2)
dt = delaunayTriangulation(t',position_realigned(:,1)*cam2cm) ;
tri = dt.ConnectivityList ; 
trisurf(tri,t',position_realigned(:,1)*cam2cm,CMP_mean) ;
shading interp ;
colorbar


