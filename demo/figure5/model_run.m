function [testIdx, Qopt2, llhd_train, llhd_test, llhd_spk, bit_spk] =...
    model_run(neuron, runInit, run_number,...
    pos, spike_counts, testProp, nknots, usr_dir, r_path, r_wd)

idx0 = 1;
spk = spike_counts(idx0:end,neuron);
T = length(spk);

% trainProp = 1 - testProp;
testIdx = randsample(T,round(T*testProp));
trainIdx = setdiff(1:T, testIdx);

spk_test = spk*NaN;
spk_train = spk*NaN;
spk_test(testIdx) = spk(testIdx);
spk_train(trainIdx) = spk(trainIdx);

Gnknots=1;
X = getCubicBSplineBasis((pos + 250)/500,nknots,false);
G = getCubicBSplineBasis((pos + 250)/500,Gnknots,false);

spk_train_narm = spk_train(~isnan(spk_train));
X_narm = X(~isnan(spk_train),:);
G_narm = G(~isnan(spk_train),:);


%% dCMP-(12,1)
nInit = find(run_number(~isnan(spk_train))>runInit,1); % fit with some min # of runs down the track
writematrix(spk_train_narm(1:nInit), [r_wd '\y.csv'])
writematrix(X_narm(1:nInit, :),[r_wd '\X.csv'])
writematrix(G_narm(1:nInit, :),[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

Q = eye(length(theta0))*1e-4;
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, spk_train', X, G,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q);

theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = [1e-4*ones(1, min(2, size(X, 2))+ min(2, size(G, 2)))];
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_na(Q, theta01, spk_train',X,G,...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

% Q_lam = Qopt(1);
Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(X, 2)-1)]; % multi
Q_nu = Qopt1(min(size(X, 2), 2)+1); % single
Qoptmatrix1 = diag([Q_lam Q_nu]);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, X,G, theta01, W01,...
    eye(length(theta01)), Qoptmatrix1, spk_train');

[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit1 = reshape(theta_newton_vec, [], T);


lam1 = zeros(T,1);
nu1 = zeros(T,1);
CMP_mean1 = zeros(T,1);
CMP_var1 = zeros(T,1);
logZ1 = zeros(T,1);

for t = 1:T
    lam1(t) = exp(X(t,:)*theta_fit1(1:(nknots+1), t));
    nu1(t) = exp(G(t,:)*theta_fit1((nknots+2):end, t));
    [CMP_mean1(t), CMP_var1(t), ~, ~, ~, logZ1(t)] = ...
            CMPmoment(lam1(t), nu1(t), 1000);
end


llhd1_train = nansum(spk_train.*log((lam1+(lam1==0))) -...
    nu1.*gammaln(spk_train + 1) - logZ1);
llhd1_test = nansum(spk_test.*log((lam1+(lam1==0))) -...
    nu1.*gammaln(spk_test + 1) - logZ1);


%% dPoi-(12)
b0 = glmfit(X_narm(1:nInit, :),...
    spk_train_narm(1:nInit)','poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
ppasmoo_poissexp_nan(spk_train,X, b0,eye(length(b0)),eye(length(b0)),1e-4*eye(length(b0)));

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(X, 2)));

f = @(Q) helper_poisson_nan(Q, theta02, spk_train,...
    X, W02, eye(length(theta02)));
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(X, 2))),QUB*ones(1, min(2, size(X, 2))), [], options);
Qoptmatrix2 = diag([Qopt2(1) Qopt2(2)*ones(1, size(X, 2)-1)]);

gradHess_tmp = @(vecTheta) gradHessTheta_Poisson_nan(vecTheta, X, theta02, W02,...
    eye(length(theta02)), Qoptmatrix2, spk_train);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit2 = reshape(theta_newton_vec, [], T);
lam2 = exp(sum(X .* theta_fit2', 2));

llhd2_train = nansum(-lam2 + log((lam2+(lam2==0))).*spk_train - gammaln(spk_train + 1));
llhd2_test = nansum(-lam2 + log((lam2+(lam2==0))).*spk_test - gammaln(spk_test + 1));

%% sCMP-(12,1)
writematrix(spk_train_narm, [r_wd '\y.csv'])
writematrix(X_narm,[r_wd '\X.csv'])
writematrix(G_narm,[r_wd '\G.csv'])

RunRcode([r_wd '\cmpRegression.r'],r_path);
theta_fit3 = readmatrix([r_wd '\cmp_t1.csv']);

lam3 = exp(X*theta_fit3(1:(nknots+1), :));
nu3 = exp(G*theta_fit3((nknots+2):end, :));
logZ3 = 0*lam3;

CMP_mean3 = 0*lam3;
CMP_var3 = 0*lam3;
for m = 1:length(lam3)
    [CMP_mean3(m), CMP_var3(m), ~, ~, ~, logZ3(m)] = ...
            CMPmoment(lam3(m), nu3(m), 1000);
end

llhd3_train = nansum(spk_train.*log((lam3+(lam3==0))) -...
    nu3.*gammaln(spk_train + 1) - logZ3);
llhd3_test = nansum(spk_test.*log((lam3+(lam3==0))) -...
    nu3.*gammaln(spk_test + 1) - logZ3);


%% sPoi - (12,1)
theta_fit4 = glmfit(X_narm,spk_train_narm,'poisson','constant','off');
lam4 = exp(X*theta_fit4);

llhd4_train = nansum(-lam4 + log((lam4+(lam4==0))).*spk_train - gammaln(spk_train + 1));
llhd4_test = nansum(-lam4 + log((lam4+(lam4==0))).*spk_test - gammaln(spk_test + 1));


%% output
lam_null = nanmean(spk_train_narm);
llhdn_train = nansum(-lam_null + log(lam_null)*spk_train - gammaln(spk_train + 1));
llhdn_test = nansum(-lam_null + log(lam_null)*spk_test - gammaln(spk_test + 1));

llhd_train = [llhd1_train llhd2_train llhd3_train llhd4_train llhdn_train];
llhd_test = [llhd1_test llhd2_test llhd3_test llhd4_test llhdn_test];

llhd_spk = diag([nansum(spk_train) nansum(spk_test)])\[llhd_train;llhd_test];
bit_spk = diag([nansum(spk_train) nansum(spk_test)])\...
    (([llhd_train(1:(end-1));llhd_test(1:(end-1))] -[llhdn_train llhdn_test]')/log(2));



end