addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
%
%%
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
neuron = 27;

%%
pos = (position_circular-1)*cam2cm;
spk = spike_counts(:,neuron);

nknots=5;
x0 = linspace(min(pos), max(pos), 500);
bas = getCubicBSplineBasis((x0-min(x0))/range(x0),nknots,false);

X = getCubicBSplineBasis((pos - min(pos))/range(pos),nknots,false);
[b,~,~] = glmfit(X,spk,'poisson','constant','off');

figure(2)
plot(pos,spk,'.')
hold on
plot(x0,exp(bas*b),'LineWidth',2)
hold off

%%

nknots=4;
Gnknots=4;
X = getCubicBSplineBasis((pos - min(pos))/range(pos),nknots,false);
G = getCubicBSplineBasis((pos - min(pos))/range(pos),Gnknots,false);

nCMP = 1000;
writematrix(spk(1:nCMP), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\y.csv')
writematrix(X(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\X.csv')
writematrix(G(1:nCMP, :), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\G.csv')

%% fit
RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmpreg.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta0 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\demo\hc\cmp_t1.csv');

windType = 'forward';

% Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-4,Gnknots,1)]); % single G
Q = diag([repmat(1e-3,nknots+1,1); repmat(1e-3,Gnknots + 1,1)]); % multi G

% fit smoother twice
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk', X, G,...
    eye(length(theta0)),eye(length(theta0)),Q, 5, windType); % initial: use window 5?

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

% window selection
winSizeSet = [1 linspace(10, 60, 6)];
np = size(X, 2) + size(G, 2);

preLL_winSize_pred = zeros(1, length(winSizeSet));
preLL_winSize_filt = zeros(1, length(winSizeSet));
preLL_winSize_smoo = zeros(1, length(winSizeSet));

idx = 1;
for k = winSizeSet
    
    [~,~,lam_pred,nu_pred,log_Zvec_pred,...
        lam_filt,nu_filt,log_Zvec_filt,...
        lam_smoo,nu_smoo,log_Zvec_smoo] =...
        ppasmoo_compoisson_v2_window_fisher(theta02, spk', X, G,...
        W02,eye(length(theta0)),Q, k, windType);
    
    
    if(length(log_Zvec_pred) == length(spk))
        preLL_winSize_pred(idx) = sum(spk'.*log((lam_pred+(lam_pred==0))) -...
            nu_pred.*gammaln(spk' + 1) - log_Zvec_pred);
        preLL_winSize_filt(idx) = sum(spk'.*log((lam_filt+(lam_filt==0))) -...
            nu_filt.*gammaln(spk' + 1) - log_Zvec_filt);
        preLL_winSize_smoo(idx) = sum(spk'.*log((lam_smoo+(lam_smoo==0))) -...
            nu_smoo.*gammaln(spk' + 1) - log_Zvec_smoo);
    else
        preLL_winSize_pred(idx) = -Inf;
        preLL_winSize_filt(idx) = -Inf;
        preLL_winSize_smoo(idx) = -Inf;
    end
    idx = idx + 1;
end

subplot(1, 3, 1)
plot(winSizeSet, preLL_winSize_pred)
title('prediction')
subplot(1, 3, 2)
plot(winSizeSet, preLL_winSize_filt)
title('filtering')
subplot(1, 3, 3)
plot(winSizeSet, preLL_winSize_smoo)
title('smoothing')

[~, winIdx] = max(preLL_winSize_filt);
optWinSize = winSizeSet(winIdx);

[theta_fit,W_fit] =...
    ppasmoo_compoisson_v2_window_fisher(theta02, spk', X, G,...
    W02,eye(length(theta0)),Q, optWinSize, windType);

save('C:\Users\gaw19004\Desktop\COM_POI_data\hc_27_noQTune.mat')

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
subplot(1, 2, 1)
hold on
plot(spk)
xlabel('steps')
ylabel('Spike Counts')
plot(CMP_mean, 'LineWidth', 2)
legend('obs.','CMP-mean')
hold off

subplot(1, 2, 2)
plot(CMP_var./CMP_mean, 'LineWidth', 2)
ylabel('Fano factor')
xlabel('steps')



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


