
% function adaptive_cmp(X,y)

% X = [ones(length(vvecd),1) vvecd' log(f(vvecd')+1)];
% fx = @(x) (x-abs(x))/2;
% X = [ones(length(vvecd),1) vvecd' fx(-vvecd')];
X = [ones(length(vvecd),1) hyp(vvecd',0.1)];
yna = y(all(isfinite(X),2));
Xna = X(all(isfinite(X),2),:);

G = [ones(length(vvecd),1)];
% G = [ones(length(vvecd),1) abs(vvecd')];
G = [ones(length(vvecd),1) exp(-abs(vvecd').^2/50.^2)];
Gna = G(all(isfinite(X),2),:);
Gna(:,2)=zscore(Gna(:,2));

% Xna(:,2:end) = zscore(Xna(:,2:end));


addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('C:\Users\ianhs\Documents\GitHub\COM_POISSON'));

% usr_dir = 'C:\Users\ianhs';
% r_path = 'C:\Program Files\R\R-4.2.1\bin';
% r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];



%% model fit

n = size(Xna,1);

% Initial fit using R
init_n = 5000; 
writematrix(yna(1:init_n), [r_wd '\y.csv'])
writematrix(Xna(1:init_n,:),[r_wd '\X.csv'])
writematrix(Gna(1:init_n,:),[r_wd '\G.csv'])
tmp=pwd();
cd(r_wd)
RunRcode([r_wd '\cmpRegression.r'],r_path);
cd(tmp)
theta0 = readmatrix([r_wd '\cmp_t1.csv']);

% Initial fit of time-varying params using default Q
Q = 1e-5*eye(length(theta0));
[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, yna', Xna, Gna,...
    10e-3*eye(length(theta0)),eye(length(theta0)),Q);

% Optimize Q using predictive log-likelihood
theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

QLB = 1e-8;
QUB = 1e-3;
Q0 = 1e-6*ones(1, min(2, size(Xna, 2))+ min(2, size(Gna, 2)));
DiffMinChange = QLB;
DiffMaxChange = QUB*0.1;
MaxFunEvals = 50;
MaxIter = 50;

f = @(Q) helper_na(Q, theta01, yna',Xna,Gna,...
    W01,eye(length(theta0)));
options = optimset('DiffMinChange',DiffMinChange,'DiffMaxChange',DiffMaxChange,...
    'MaxFunEvals', MaxFunEvals, 'MaxIter', MaxIter);
Qopt = fmincon(f,Q0,[],[],[],[],...
    QLB*ones(1, length(Q0)),QUB*ones(1, length(Q0)), [], options);


% Fit theta with optimized Q
Q_lam = [Qopt(1) repmat(Qopt(2),1,size(Xna,2)-1)];
Q_nu = [Qopt(3)];
if length(Qopt)>3
    Q_nu = [Q_nu repmat(Qopt(4),1,size(Gna,2)-1)];
end
Qoptmatrix = diag([Q_lam Q_nu]);

[theta_fit_tmp,W_fit_tmp] =...
    ppasmoo_compoisson_fisher_na(theta0, yna', Xna, Gna,...
    eye(length(theta0)),eye(length(theta0)),Qoptmatrix);
theta01 = theta_fit_tmp(:, 1);
W01 = W_fit_tmp(:, :, 1);

gradHess_tmp = @(vecTheta) gradHessTheta_na(vecTheta, Xna,Gna,...
    theta01, W01,eye(length(theta01)), Qoptmatrix, yna');
[theta_newton_vec,~,hess_tmp,~] = newtonGH(gradHess_tmp,theta_fit_tmp(:),1e-10,1000);
theta_fit = reshape(theta_newton_vec, [], n);


% Calculate moments
lam_fit = zeros(1, n);
nu_fit = zeros(1, n);
CMP_mean_fit = zeros(1, n);
CMP_var_fit = zeros(1, n);
logZ_fit = zeros(1,n);
maxy=2*max(yna);
cmp_pdf = zeros(maxy+1,n);
for m = 1:n
    lam_fit(m) = exp(Xna(m,:)*theta_fit(1:size(Xna,2),m));
    nu_fit(m) = exp(Gna(m,:)*theta_fit((size(Xna,2)+1):end,m));
    [CMP_mean_fit(m), CMP_var_fit(m), ~, ~, ~, logZ_fit(m)] = ...
            CMPmoment(lam_fit(m), nu_fit(m), 1000);
        
    cmp_pdf(:,m) = com_pdf(0:maxy, lam_fit(m), nu_fit(m));
end
CMP_ff_fit = CMP_var_fit./CMP_mean_fit;
