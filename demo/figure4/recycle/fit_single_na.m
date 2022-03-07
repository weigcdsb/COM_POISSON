addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

%% samples

% full
trial_x_full=repmat(theta',size(data.EVENTS,2),1);
trial_y_full=[];
c=1;
stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y_full(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
        end
        t=t+stim_length;
        c=c+1;
    end
end

% sub
rng(1)

nAll = length(theta);
nSS = round(nAll/2);

ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
for k = 1:size(data.EVENTS, 2)
    ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
        sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
end

trial_y = trial_y_full*nan;
trial_y(ssIdx,:) = trial_y_full(ssIdx,:);
trial_y_narm = trial_y_full(ssIdx, :);

neuron=5;
nplot = 10;

[~,theta_idx]=sort(theta);

smoothing=5;
mff=[]; mmm=[];
obs_ff = []; obs_mean = [];
for i=1:size(data.EVENTS, 2)-smoothing
    
    c=1;
    for j = theta_idx
        tid = [j:nAll:(nAll*smoothing)]+i*nAll;
        ff = getFF(trial_y_full(tid, neuron));
        obs_ff(c, i) = ff;
        obs_mean(c, i) = mean(trial_y_full(tid, neuron));
        c = c+1;
    end
    
    ff=[];
    c=1;
    for stim=linspace(0,2*pi,nplot)
        tv=[];
        tv = find(abs(theta-stim)<pi/5);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):nAll:(nAll*smoothing)]+i*nAll];
        end
        ff = getFF(trial_y_full(tid,neuron));
        mff(c,i) = mean(ff);
        mmm(c,i) = mean(trial_y_full(tid,neuron));
        c=c+1;
    end
end

figure(1)
subplot(2,2,1)
plot([1:(120-smoothing)]+smoothing/2,mmm')
xlabel('Trial')
ylabel('Mean')
subplot(2,2,2)
plot([1:(120-smoothing)]+smoothing/2,mff')
xlabel('Trial')
ylabel('Fano Factor')
subplot(2,2,3)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
colorbar()
title('Mean-obs')
subplot(2,2,4)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
title('Fano Factor')
colorbar()




spk_vec = trial_y(:,neuron)';
spk_vec_narm = trial_y_narm(:, neuron)';
T = length(trial_x_full);

nknots = 5;


%% fit 1: adaptive CMP, nBasis for lambda = 5, nBasis for nu = 3
Gnknots_full = 3;

Xb = getCubicBSplineBasis(trial_x_full,nknots,true);
Gb_full = getCubicBSplineBasis(trial_x_full,Gnknots_full,true);

Xb_narm = Xb(ssIdx,:);
Gb_full_narm = Gb_full(ssIdx,:);

initIdx = max(10*nSS, find(cumsum(spk_vec_narm) > 200, 1, 'first'));
writematrix(spk_vec_narm(1:initIdx)', [r_wd '\y.csv'])
writematrix(Xb_narm(1:initIdx, :),[r_wd '\X.csv'])
writematrix(Gb_full_narm(1:initIdx, :),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-4,Gnknots_full + 1,1)]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec, Xb, Gb_full,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb_full, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
nSub = round(size(trial_y, 1));

f = @(Q) helper_na(Q, theta01, trial_y(1:nSub,neuron)',Xb,Gb_full,...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb_full, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, Xb,Gb_full, theta01, W01,...
    eye(length(theta01)), Qoptmatrix1, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit1 = reshape(theta_newton_vec, [], T);

lam1 = zeros(1,T);
nu1 = zeros(1,T);
CMP_mean1 = zeros(1, T);
CMP_var1 = zeros(1, T);
logZ1 = zeros(1,T);

for m = 1:T
    lam1(m) = exp(Xb(m,:)*theta_fit1(1:(nknots+1), m));
    nu1(m) = exp(Gb_full(m,:)*theta_fit1((nknots+2):end, m));
    [CMP_mean1(m), CMP_var1(m), ~, ~, ~, logZ1(m)] = ...
            CMPmoment(lam1(m), nu1(m), 1000);
end

CMP_ff1 = CMP_var1./CMP_mean1;

x0 = linspace(0,2*pi,256);
basX = getCubicBSplineBasis(x0,nknots,true);
basG_full = getCubicBSplineBasis(x0,Gnknots_full,true);
[mean1, var1, ff1] = cmp_grid(theta_fit1(:,1:50:T), nknots, basX, basG_full, 1000);


figure(3)
subplot(1,2,1)
imagesc(1:size(data.EVENTS, 2),x0, mean1)
ylabel('Stimulus Direction [rad]')
title('Mean-fit')
colorbar()
subplot(1,2,2)
imagesc(1:size(data.EVENTS, 2),x0, ff1)
title('FF-fit')
colorbar()
xlabel('Trial')
