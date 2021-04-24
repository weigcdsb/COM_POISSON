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
neuron=24; % 24 25 42
ry = reshape(trial_y(:,neuron),100,[]);

nknots=5;
X = getCubicBSplineBasis(theta',nknots,true);
x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

basData = bas(:,2:end);
% writematrix(ry,'D:\GitHub\COM_POISSON\runRcode\ry.csv')
% writematrix(X(:,2:end),'D:\GitHub\COM_POISSON\runRcode\X.csv')
% writematrix(basData,'D:\GitHub\COM_POISSON\runRcode\basData.csv')
% writematrix(theta','D:\GitHub\COM_POISSON\runRcode\theta.csv')

trial = 1;
fitData = [ry(:, trial) X(:,2:end)];
writematrix(fitData,'D:\GitHub\COM_POISSON\runRcode\fitData.csv')

%% fit the data

RunRcode('D:\GitHub\COM_POISSON\runRcode\cmpRegression.r',...
    'E:\software\R\R-4.0.2\bin');

theta0 = readmatrix('cmp_t1.csv');
theta0 = theta0(:, 2:end);
theta0 = theta0(:);

% [theta_fit1,W_fit1] =...
%     ppafilt_compoisson_v3(theta0, ry,repmat(X, [1 1 size(ry, 2)]),repmat(X, [1 1 size(ry, 2)]),...
%     eye(length(theta0)),eye(length(theta0)),1e-3*eye(length(theta0)));

[theta_fit2,W_fit2] =...
    ppasmoo_compoisson_v3(theta0, ry,repmat(X, [1 1 size(ry, 2)]),repmat(X, [1 1 size(ry, 2)]),...
    eye(length(theta0)),eye(length(theta0)),1e-3*eye(length(theta0)));

%%
plot(theta_fit2(1:(nknots+1), :)')
plot(theta_fit2((nknots+2):end, :)')

%% calculate mean & var

theta_fit = theta_fit2;
x02 = linspace(0,2*pi,10);
bas2 = getCubicBSplineBasis(x02,nknots,true);

lam = exp(bas2 * theta_fit(1:(nknots+1), :));
nu = exp(bas2 * theta_fit((nknots+2):end, :));

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

smoothing=20;
mff=[]; mmm=[];
for i=1:(120-smoothing)
    ff=[];
    c=1;
    for stim=x02
        tv=[];
        tv = find(abs(theta-stim)<pi/5);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):100:(100*smoothing)]+i*100];
        end
        ff = getFF(trial_y(tid,neuron));
        mff(c,i) = mean(ff);
        mmm(c,i) = mean(trial_y(tid,neuron));
        c=c+1;
    end
end


subplot(2,2,1)
plot(CMP_mean')
subplot(2,2,2)
plot(CMP_ff')
subplot(2,2,3)
plot([1:(120-smoothing)]+smoothing/2,mmm')
xlabel('Trial')
ylabel('Mean')
subplot(2,2,4)
plot([1:(120-smoothing)]+smoothing/2,mff')
xlabel('Trial')
ylabel('Fano Factor')


