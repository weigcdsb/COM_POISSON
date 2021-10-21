function [llhd_per_spk, bit_per_spk] = models_run(neuron, trial_x_full, trial_y_full,...
    ssIdx, nSS, nAll, constNu, usr_dir, r_path)

% to debug
% neuron = n;
% ssIdx = SSIDX{n};
% constNu = false;


trial_x_ss = trial_x_full(ssIdx);
trial_y_ss = trial_y_full(ssIdx, :);

trial_x = trial_x_ss;
trial_y = trial_y_ss;
nObs = nSS;
T = length(trial_x);

spk_vec = trial_y(:,neuron)';
ry = reshape(trial_y(:,neuron),nObs,[]);
nknots = 5;

%% fit 1: adaptive CMP, nBasis for lambda = 5, nBasis for nu = 3

Gnknots_full = 3;

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb_full = getCubicBSplineBasis(trial_x,Gnknots_full,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb_full((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

windType = 'forward';
RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);


% Q-tune
Q = diag([repmat(1e-4,nknots+1,1); repmat(1e-4,Gnknots_full + 1,1)]);
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb_full,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q, 1, windType);

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

f = @(Q) helper_window_v2(Q, theta01, trial_y(1:nSub,neuron)',Xb,Gb_full,...
    W01,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt1 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt1(1) Qopt1(2)*ones(1, size(Xb, 2)-1)];
Q_nu = [Qopt1(3) Qopt1(4)*ones(1, size(Gb_full, 2) - 1)];
Qoptmatrix1 = diag([Q_lam Q_nu]);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, Xb,Gb_full, theta01, W01,...
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

Xb = getCubicBSplineBasis(trial_x,nknots,true);
Gb = getCubicBSplineBasis(trial_x,Gnknots,true);

trial = 1:15;
writematrix(reshape(ry(:, trial), [], 1), [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\y.csv'])
writematrix(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\X.csv'])
writematrix(Gb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    [usr_dir '\Documents\GitHub\COM_POISSON\runRcode\G.csv'])

windType = 'forward';
RunRcode([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmpRegression.r'],r_path);
theta0 = readmatrix([usr_dir '\Documents\GitHub\COM_POISSON\runRcode\cmp_t1.csv']);


% Q-tune
Q = eye(length(theta0))*1e-4;
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_v2_window_fisher(theta0, spk_vec, Xb, Gb,...
    eye(length(theta0))*1e-1,eye(length(theta0)),Q, 1, windType);

theta02 = theta_fit_tmp(:, 1);
W02 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2))+ min(2, size(Gb, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 500;
MaxIter = 500;

f = @(Q) helper_window_v2(Q, theta02, trial_y(1:nSub,neuron)',Xb,Gb,...
    W02,eye(length(theta0)),1, windType);
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt2 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);

Q_lam = [Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)];
Q_nu = Qopt2(3);
Qoptmatrix2 = diag([Q_lam Q_nu]);


gradHess_tmp = @(vecTheta) gradHessTheta(vecTheta, Xb,Gb, theta02, W02,...
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

%% fit 3: adaptive CMP, constant nu
if constNu
    trial = 1:15;
    Xb = getCubicBSplineBasis(trial_x,nknots,true);
    Qopt3 = diag([Qopt2(1) Qopt2(2)*ones(1, size(Xb, 2)-1)]);
    
    iterMax = 1000;
    nu_trace = ones(iterMax, 1);
    theta_lam = ones(size(Xb, 2),T, iterMax);
    [b,~,~] = glmfit(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
        reshape(ry(:, trial), [], 1),'poisson','constant','off');
    theta_lam(:,:,1) =repmat(b, 1, T);
    smoo_flag = 1;
    
    for g = 2:iterMax
        
        % (1) update lambda_i
        if smoo_flag
            try
                [theta_lam(:,:,g),~,~] = ppasmoo_cmp_fixNu(spk_vec,Xb,nu_trace(g-1),...
                    theta_lam(:,1,g-1),eye(size(Xb, 2))*1e-1,eye(size(Xb, 2)),Qopt3);
            catch
                theta_tmp = theta_lam(:,:,g-1);
                gradHess_tmp = @(vecTheta) gradHessTheta_CMP_fixNu(vecTheta, Xb,...
                    nu_trace(g-1),theta_lam(:,1,g-1),eye(size(Xb, 2))*1e-1,...
                    eye(size(Xb, 2)), Qopt3, spk_vec);
                [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
                theta_lam(:,:,g)  = reshape(theta_newton_vec, [], T);
            end
        else
            theta_tmp = theta_lam(:,:,g-1);
            gradHess_tmp = @(vecTheta) gradHessTheta_CMP_fixNu(vecTheta, Xb,...
                nu_trace(g-1),theta_lam(:,1,g-1),eye(size(Xb, 2))*1e-1,...
                eye(size(Xb, 2)), Qopt3, spk_vec);
            [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
            if(sum(isnan(theta_newton_vec)) ~= 0)
                disp('use smoother')
                [theta_tmp,~,~] = ppasmoo_cmp_fixNu(spk_vec,Xb,nu_trace(g-1),...
                    theta_lam(:,1,g-1),eye(size(Xb, 2))*1e-1,eye(size(Xb, 2)),Qopt3);
                [theta_newton_vec,~,~,~] = newtonGH(gradHess_tmp,theta_tmp(:),1e-10,1000);
            end
            theta_lam(:,:,g)  = reshape(theta_newton_vec, [], T);
        end
        
        if(norm(theta_lam(:,:,g) - theta_lam(:,:,g-1), 'fro') < sqrt(1e-2*size(Xb, 2)*T))
            disp("normXdiff: " + norm(theta_lam(:,:,g) - theta_lam(:,:,g-1), 'fro'))
            smoo_flag = 0;
        end
        
        % (2) update nu
        gradHess_tmp = @(nu) gradHessNu(nu, Xb, theta_lam(:,:,g), spk_vec);
        nu_trace(g) = newtonGH(gradHess_tmp, nu_trace(g-1), 1e-10, 1000);
        
        disp(norm(nu_trace(g) - nu_trace(g-1)))
        
        if(norm(nu_trace(g) - nu_trace(g-1)) < 1e-2)
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
    
    
end


%% fit 4: adaptive Poisson
trial = 1:15;
b0 = glmfit(Xb((nObs*(trial(1)-1)+1):nObs*trial(end), :),...
    reshape(ry(:, trial), [], 1),'poisson','constant','off');

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_poissexp(spk_vec,Xb, b0,eye(length(b0))*1e-1,eye(length(b0)),1e-4*eye(length(b0)));

theta04 = theta_fit_tmp(:, 1);
W04 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-4*ones(1, min(2, size(Xb, 2)));

f = @(Q) helper_poisson(Q, theta04, trial_y(1:nSub,neuron)',...
    Xb, W04, eye(length(theta04)));
Qopt4 = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, min(2, size(Xb, 2))),QUB*ones(1, min(2, size(Xb, 2))), [], options);
if(sum(isnan(Qopt4)) > 0)
    error('error')
end
Qoptmatrix4 = diag([Qopt4(1) Qopt4(2)*ones(1, size(Xb, 2)-1)]);



gradHess_tmp = @(vecTheta) gradHessTheta_Poisson(vecTheta, Xb, theta04, W04,...
    eye(length(theta04)), Qoptmatrix4, spk_vec);
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit4 = reshape(theta_newton_vec, [], T);
lam4 = exp(sum(Xb .* theta_fit4', 2));

%% fit 5: static CMP

writematrix(spk_vec', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full, 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

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

%% fit 6: static Poisson
theta_fit6 = glmfit(Xb,spk_vec,'poisson','constant','off');
lam6 = exp(Xb*theta_fit6);


%% training set llhd
llhd1 = sum(spk_vec.*log((lam1+(lam1==0))) -...
    nu1.*gammaln(spk_vec + 1) - logZ1);
llhd2 = sum(spk_vec.*log((lam2+(lam2==0))) -...
    nu2.*gammaln(spk_vec + 1) - logZ2);

if constNu
    llhd3 = sum(spk_vec.*log((lam3+(lam3==0))) -...
        nu3*gammaln(spk_vec + 1) - logZ3);
end

llhd4 = sum(-lam4' + log((lam4'+(lam4'==0))).*spk_vec - gammaln(spk_vec + 1));
llhd5 = sum(spk_vec'.*log((lam5+(lam5==0))) -...
    nu5.*gammaln(spk_vec' + 1) - logZ5);
llhd6 = sum(-lam6' + log((lam6'+(lam6'==0))).*spk_vec - gammaln(spk_vec + 1));

%% held-out set... linear interpolation
hoIdx = setdiff(1:length(trial_x_full), ssIdx);
nHo = nAll - nObs;
trial_x_ho = trial_x_full(hoIdx);
spk_vec_ho = trial_y_full(hoIdx,neuron)';
T_ho = length(hoIdx);

Xb_ho = getCubicBSplineBasis(trial_x_ho,nknots,true);
Gb_ho_full = getCubicBSplineBasis(trial_x_ho,Gnknots_full,true);
Gb_ho = getCubicBSplineBasis(trial_x_ho,Gnknots,true);

theta_ho1 = theta_interp1(theta_fit1, ssIdx, hoIdx);
theta_ho2 = theta_interp1(theta_fit2, ssIdx, hoIdx);
theta_ho4 = theta_interp1(theta_fit4, ssIdx, hoIdx);


[lam1_ho, nu1_ho, logZ1_ho, CMP_mean1_ho, CMP_var1_ho] =...
    CMP_seq_calc(theta_ho1, Xb_ho, Gb_ho_full, nknots, 1000);
[lam2_ho, nu2_ho, logZ2_ho, CMP_mean2_ho, CMP_var2_ho] =...
    CMP_seq_calc(theta_ho2, Xb_ho, Gb_ho, nknots, 1000);

if constNu
    theta_ho3 = theta_interp1(theta_fit3, ssIdx, hoIdx);
    lam3_ho = zeros(1,T);
    logZ3_ho = zeros(1,T);
    CMP_mean3_ho = zeros(1,T);
    CMP_var3_ho = zeros(1,T);
    for t = 1:T
        lam3_ho(t) = exp(Xb_ho(t,:)*theta_ho3(:, t));
        [CMP_mean3_ho(t), CMP_var3_ho(t), ~, ~, ~, logZ3_ho(t)] = ...
            CMPmoment(lam3_ho(t), nu3, 1000);
    end
end


lam4_ho = exp(sum(Xb_ho .* theta_ho4', 2));
[lam5_ho, nu5_ho, logZ5_ho, CMP_mean5_ho, CMP_var5_ho] =...
    CMP_seq_calc(repmat(theta_fit5,1,T_ho), Xb_ho, Gb_ho_full, nknots, 1000);
lam6_ho = exp(Xb_ho*theta_fit6);

llhd1_ho = sum(spk_vec_ho.*log((lam1_ho+(lam1_ho==0))) -...
    nu1_ho.*gammaln(spk_vec_ho + 1) - logZ1_ho);
llhd2_ho = sum(spk_vec_ho.*log((lam2_ho+(lam2_ho==0))) -...
    nu2_ho.*gammaln(spk_vec_ho + 1) - logZ2_ho);

if constNu
    llhd3_ho = sum(spk_vec_ho.*log((lam3_ho+(lam3_ho==0))) -...
        nu3*gammaln(spk_vec_ho + 1) - logZ3_ho);
end

llhd4_ho = sum(-lam4_ho' + log((lam4_ho'+(lam4_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));
llhd5_ho = sum(spk_vec_ho.*log((lam5_ho+(lam5_ho==0))) -...
    nu5_ho.*gammaln(spk_vec_ho + 1) - logZ5_ho);
llhd6_ho = sum(-lam6_ho' + log((lam6_ho'+(lam6_ho'==0))).*spk_vec_ho - gammaln(spk_vec_ho + 1));


%% llhd per spike

% null model
writematrix(spk_vec', 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\yAll.csv')
writematrix(Xb(:,1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\XAll.csv')
writematrix(Gb_full(:,1), 'C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\GAll.csv')

RunRcode('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmpRegression_all.r',...
    'C:\Users\gaw19004\Documents\R\R-4.0.2\bin');
theta_null = readmatrix('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\runRcode\cmp_all.csv');
lam_null = exp(theta_null(1));
nu_null = exp(theta_null(2));
[CMP_mean_null, CMP_var_null, ~, ~, ~, logZ_null] = ...
    CMPmoment(lam_null, nu_null, 1000);

llhd_null = sum(spk_vec'*log((lam_null+(lam_null==0))) -...
    nu_null*gammaln(spk_vec' + 1) - logZ_null);
llhd_null_ho = sum(spk_vec_ho*log((lam_null+(lam_null==0))) -...
    nu_null*gammaln(spk_vec_ho + 1) - logZ_null);

if constNu
    llhd_per_spk = diag([sum(spk_vec) sum(spk_vec_ho)])\...
        [llhd1 llhd2 llhd3 llhd4 llhd5 llhd6 llhd_null;...
        llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho llhd_null_ho];
    
    %bit/spks
    bit_per_spk = diag([sum(spk_vec) sum(spk_vec_ho)])\...
        (([llhd1 llhd2 llhd3 llhd4 llhd5 llhd6;...
        llhd1_ho llhd2_ho llhd3_ho llhd4_ho llhd5_ho llhd6_ho] -...
        [llhd_null llhd_null_ho]')/log(2));
    
else
    llhd_per_spk = diag([sum(spk_vec) sum(spk_vec_ho)])\...
        [llhd1 llhd2 llhd4 llhd5 llhd6 llhd_null;...
        llhd1_ho llhd2_ho llhd4_ho llhd5_ho llhd6_ho llhd_null_ho];
    
    %bit/spks
    bit_per_spk = diag([sum(spk_vec) sum(spk_vec_ho)])\...
        (([llhd1 llhd2 llhd4 llhd5 llhd6;...
        llhd1_ho llhd2_ho llhd4_ho llhd5_ho llhd6_ho] -...
        [llhd_null llhd_null_ho]')/log(2));


end












end