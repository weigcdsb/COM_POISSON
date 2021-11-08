addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
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

neuron=13;
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
writematrix(spk_vec_narm(1:initIdx)', [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb_narm(1:initIdx, :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb_full_narm(1:initIdx, :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);

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

%% fit 2: adaptive CMP, nBasis for nu = 1
Gnknots=1;

Xb = getCubicBSplineBasis(trial_x_full,nknots,true);
Gb = getCubicBSplineBasis(trial_x_full,Gnknots,true);
Xb_narm = Xb(ssIdx,:);
Gb_narm = Gb(ssIdx,:);

writematrix(spk_vec_narm(1:initIdx)', [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb_narm(1:initIdx, :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb_narm(1:initIdx, :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);

Q = eye(length(theta0))*1e-4;
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;
nSub = round(size(trial_y, 1));

f = @(Q) helper_na(Q, theta02, trial_y(1:nSub,neuron)',Xb,Gb,...
    W02,eye(length(theta0)));
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)];
Q_nu = Qopt2(3);
Qoptmatrix2 = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, Xb,Gb, theta02, W02,...
    eye(length(theta02)), Qoptmatrix2, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], T);

lam2 = zeros(1, T);
nu2 = zeros(1, T);
CMP_mean2 = zeros(1, T);
CMP_var2 = zeros(1, T);
logZ2 = zeros(1,T);

for m = 1:T
    lam2(m) = exp(Xb(m,:)*theta_fit2(1:(nknots+1), m));
    nu2(m) = exp(Gb(m,:)*theta_fit2((nknots+2):end, m));
    [CMP_mean2(m), CMP_var2(m), ~, ~, ~, logZ2(m)] = ...
            CMPmoment(lam2(m), nu2(m), 1000);
end

CMP_ff2 = CMP_var2./CMP_mean2;

%% fit 4: adaptive Poisson
initIdx = max(15*nSS, find(cumsum(spk_vec_narm) > 200, 1, 'first'));
b0 = glmfit(Xb_narm(1:initIdx, :),...
    spk_vec_narm(1:initIdx)','poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp_nan(spk_vec,Xb, b0,eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));

theta04 = theta_fit_tmp(:, 1);
W04 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));

f = @(Q) helper_poisson_nan(Q, theta04, trial_y(1:nSub,neuron)',...
    Xb, W04, eye(length(theta04)));
Qopt4 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(Xb, 2))),QUB*ones(1, min(2, size(Xb, 2))), [], options);
Qoptmatrix4 = diag([Qopt4(1) Qopt4(2)*ones(1, size(Xb, 2)-1)]);

gradHess_tmp = @(vecTheta) gradHessTheta_Poisson_nan(vecTheta, Xb, theta04, W04,...
    eye(length(theta04)), Qoptmatrix4, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit4 = reshape(theta_newton_vec, [], T);
lam4 = exp(sum(Xb .* theta_fit4', 2));

%% fit 3: adaptive CMP, constant nu
Qoptmatrix3 = Qoptmatrix4;

iterMax = 1000;
nu_trace = ones(iterMax, 1);
theta_lam = ones(size(Xb, 2),T, iterMax);
theta_lam(:,:,1) = theta_fit2(1:size(Xb, 2), :);
nu_trace(1) = mean(nu2);

smoo_flag = 1;

for g = 2:iterMax
    % (1) update lambda_i
    if smoo_flag
        [theta_lam(:,:,g),~,~] = ppasmoo_cmp_fixNu_na(spk_vec,Xb,nu_trace(g-1),...
            theta02(1:(end-1)),W02(1:(end-1), 1:(end-1)),eye(size(Xb, 2)),Qoptmatrix3);
    else
        theta_tmp = theta_lam(:,:,g-1);
        gradHess_tmp = @(vecTheta) gradHessTheta_CMP_fixNu_na(vecTheta, Xb,...
            nu_trace(g-1),theta02(1:(end-1)),W02(1:(end-1), 1:(end-1)),...
            eye(size(Xb, 2)), Qoptmatrix3, spk_vec);
        [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
        if(sum(isnan(theta_newton_vec)) ~= 0)
            disp('use smoother')
            [theta_tmp,~,~] = ppasmoo_cmp_fixNu_na(spk_vec,Xb,nu_trace(g-1),...
            theta02(1:(end-1)),W02(1:(end-1), 1:(end-1)),eye(size(Xb, 2)),Qoptmatrix3);
            [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
        end
        theta_lam(:,:,g)  = reshape(theta_newton_vec, [], T);
    end
    
    if(norm(theta_lam(:,:,g) - theta_lam(:,:,g-1), 'fro') < sqrt(1e-2*size(Xb, 2)*T))
        disp("normXdiff: " + norm(theta_lam(:,:,g) - theta_lam(:,:,g-1), 'fro'))
       smoo_flag = 0; 
    end
    
    % (2) update nu
    fun = @(nu) -llhdCMP_consNu_na(nu, Xb, theta_lam(:,:,g), spk_vec);
    nu_trace(g) = fmincon(fun,nu_trace(g-1),[],[],[],[],0,[], [], options);
    
    disp(norm(nu_trace(g) - nu_trace(g-1)))
    
    if(norm(nu_trace(g) - nu_trace(g-1)) < 5*1e-4)
        break;
    end
end

theta_fit3 = theta_lam(:,:,g);
lam3 = exp(sum(Xb .* theta_fit3', 2))';
nu3 = nu_trace(g);
CMP_mean3 = zeros(1, T);
CMP_var3 = zeros(1, T);
logZ3 = zeros(1,T);

for m = 1:T
    [CMP_mean3(m), CMP_var3(m), ~, ~, ~, logZ3(m)] = ...
            CMPmoment(lam3(m), nu3, 1000);
end

CMP_ff3 = CMP_var3./CMP_mean3;

%% fit 5: static CMP(5,3)
writematrix(spk_vec_narm', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb_narm, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full_narm, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_fit5 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');

lam5 = exp(Xb*theta_fit5(1:(nknots+1), :));
nu5 = exp(Gb_full*theta_fit5((nknots+2):end, :));
logZ5 = 0*lam5;

CMP_mean5 = 0*lam5;
CMP_var5 = 0*lam5;

for m = 1:length(lam5)
    logcum_app  = logsum_calc(lam5(m), nu5(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ5(m) = log_Z;
    CMP_mean5(m) = exp(log_A - log_Z);
    CMP_var5(m) = exp(log_B - log_Z) - CMP_mean5(m)^2;
    
end

CMP_ff5 = CMP_var5./CMP_mean5;

%% fit 6: static CMP(5,1)
writematrix(spk_vec_narm', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb_narm, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full_narm(:,1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_fit6 = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');

lam6 = exp(Xb*theta_fit6(1:(nknots+1), :));
nu6 = exp(Gb_full(:,1)*theta_fit6((nknots+2):end, :));
logZ6 = 0*lam6;

CMP_mean6 = 0*lam6;
CMP_var6 = 0*lam6;

for m = 1:length(lam6)
    logcum_app  = logsum_calc(lam6(m), nu6(m), 1000);
    log_Z = logcum_app(1);
    log_A = logcum_app(2);
    log_B = logcum_app(3);
    
    logZ6(m) = log_Z;
    CMP_mean6(m) = exp(log_A - log_Z);
    CMP_var6(m) = exp(log_B - log_Z) - CMP_mean6(m)^2;
    
end

CMP_ff6 = CMP_var6./CMP_mean6;

%% fit 7: static Poisson
theta_fit7 = glmfit(Xb_narm,spk_vec_narm,'poisson','constant','off');
lam7 = exp(Xb*theta_fit7);


%%
llhd1 = nansum(spk_vec.*log((lam1+(lam1==0))) -...
        nu1.*gammaln(spk_vec + 1) - logZ1); % dCMP(5,3)
llhd2 = nansum(spk_vec.*log((lam2+(lam2==0))) -...
        nu2.*gammaln(spk_vec + 1) - logZ2); % dCMP(5,1)

llhd3 = nansum(spk_vec.*log((lam3+(lam3==0))) -...
        nu3*gammaln(spk_vec + 1) - logZ3); % dCMP(5,cons)
    
llhd4 = nansum(-lam4' + log((lam4'+(lam4'==0))).*spk_vec - gammaln(spk_vec + 1)); %dPoi(5)
llhd5 = nansum(spk_vec'.*log((lam5+(lam5==0))) -...
        nu5.*gammaln(spk_vec' + 1) - logZ5); %sCMP(5,3)
llhd6 = nansum(spk_vec'.*log((lam6+(lam6==0))) -...
        nu6.*gammaln(spk_vec' + 1) - logZ6); % sCMP(5,1)
llhd7 = nansum(-lam7' + log((lam7'+(lam7'==0))).*spk_vec - gammaln(spk_vec + 1)); % sPoi(5)

%%
hoIdx = setdiff(1:length(trial_x_full), ssIdx);
spk_vec_ho = ones(1,size(trial_y_full,1))*nan; 
spk_vec_ho(hoIdx) = trial_y_full(hoIdx,neuron);

llhd1_ho = nansum(spk_vec_ho.*log((lam1+(lam1==0))) -...
        nu1.*gammaln(spk_vec_ho + 1) - logZ1); % dCMP(5,3)
llhd2_ho = nansum(spk_vec_ho.*log((lam2+(lam2==0))) -...
        nu2.*gammaln(spk_vec_ho + 1) - logZ2); % dCMP(5,1)

llhd3_ho = nansum(spk_vec_ho.*log((lam3+(lam3==0))) -...
        nu3*gammaln(spk_vec_ho + 1) - logZ3); % dCMP(5,cons)
    
llhd4_ho = nansum(-lam4' + log((lam4'+(lam4'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1)); %dPoi(5)
llhd5_ho = nansum(spk_vec_ho'.*log((lam5+(lam5==0))) -...
        nu5.*gammaln(spk_vec_ho' + 1) - logZ5); %sCMP(5,3)
llhd6_ho = nansum(spk_vec_ho'.*log((lam6+(lam6==0))) -...
        nu6.*gammaln(spk_vec_ho' + 1) - logZ6); % sCMP(5,1)
llhd7_ho = nansum(-lam7' + log((lam7'+(lam7'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1)); % sPoi(5)

%%
lam_null = nanmean(spk_vec);
llhdn = nansum(-lam_null + log(lam_null)*spk_vec - gammaln(spk_vec + 1));
llhdn_ho = nansum(-lam_null + log(lam_null)*spk_vec_ho - gammaln(spk_vec_ho + 1));

%%
llhd = [llhd1 llhd2 llhd3 llhd4 llhd5 llhd6 llhd7 llhdn];
llhd_ho = [llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho llhd7_ho llhdn_ho];


llhd_spk = diag([nansum(spk_vec) nansum(spk_vec_ho)])\...
    [llhd1 llhd2 llhd3 llhd4 llhd5 llhd6 llhd7 llhdn;...
    llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho llhd7_ho llhdn_ho];

bit_spk = diag([nansum(spk_vec) nansum(spk_vec_ho)])\...
    (([llhd1 llhd2 llhd3 llhd4 llhd5 llhd6 llhd7;...
    llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho llhd7_ho] -...
    [llhdn llhdn_ho]')/log(2));

["dCMP(5,3)" "dCMP(5,1)" "dCMP(5,cons)" "dPoi(5)" "sCMP(5,3)" "sCMP(5,1)" "sPoi(5)"; bit_spk]


% bit/trial
bit_trial = diag([size(data.EVENTS,2) size(data.EVENTS,2)])\...
    (([llhd1 llhd2 llhd3 llhd4 llhd5 llhd6 llhd7;...
    llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho llhd7_ho] -...
    [llhdn llhdn_ho]')/log(2));

["dCMP(5,3)" "dCMP(5,1)" "dCMP(5,cons)" "dPoi(5)" "sCMP(5,3)" "sCMP(5,1)" "sPoi(5)"; bit_trial]

%%
x0 = linspace(0,2*pi,256);
basX = getCubicBSplineBasis(x0,nknots,true);
basG_full = getCubicBSplineBasis(x0,Gnknots_full,true);
[mean1, var1, ff1] = cmp_grid(theta_fit1(:,1:50:T), nknots, basX, basG_full, 1000);




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


figure(3)
subplot(2,2,1)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_mean)
colorbar()
title('Mean-obs')
subplot(2,2,2)
imagesc([1 (120-smoothing)]+smoothing/2, [min(theta) max(theta)], obs_ff)
title('FF-obs')
colorbar()
subplot(2,2,3)
imagesc(mean1)
title('Mean-fit')
colorbar()
subplot(2,2,4)
imagesc(ff1)
title('FF-fit')
colorbar()




% 
% figure(3)
% subplot(1,2,1)
% imagesc(mean1)
% title('Mean-fit')
% colorbar()
% subplot(1,2,2)
% imagesc(ff1)
% title('FF-fit')
% colorbar()
