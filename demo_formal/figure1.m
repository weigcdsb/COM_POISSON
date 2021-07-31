addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));

%% true underlying mean & FF
nknots = 10;
x0 = linspace(0,2*pi,100);
basX = getCubicBSplineBasis(x0,nknots,true);
basG = getCubicBSplineBasis(x0,nknots,true);
T = 100;
dt = 0.5;
kStep = T/dt;


gam = zeros(size(basX, 2), kStep);
gam(4,:) = linspace(-2, .5, kStep);
gam(9,:) = linspace(-2, 1.5, kStep);
nu = exp(basG*gam);
% plot(nu)

% target means for 2 peaks
mean1 = [linspace(2, 10, round(kStep/2)) ones(1,kStep - round(kStep/2))*10];
mean2 = [linspace(2, 20, round(3*kStep/4)) ones(1,kStep - round(3*kStep/4))*20];

% find positions for 2 peaks
[~,idx1] = max(basG(:, 4));
[~,idx2] = max(basG(:, 9));

% match
beta = zeros(size(basX, 2), kStep);
logLam1 = nu(idx1, :).*log(mean1 + (nu(idx1, :) - 1)./ (2*nu(idx1, :)));
beta(4,:) = logLam1/basX(idx1, 4);
logLam2 = nu(idx2, :).*log(mean2 + (nu(idx2, :) - 1)./ (2*nu(idx2, :)));
beta(9,:) = logLam2/basX(idx2, 9);
lam = exp(basX*beta);
% plot(lam)

CMP_mean = zeros(size(lam));
CMP_var = zeros(size(lam));

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        logcum_app = logsum_calc(lam(m,n), nu(m,n), 100);
        log_Z = logcum_app(1);
        log_A = logcum_app(2);
        log_B = logcum_app(3);
        
        CMP_mean(m,n) = exp(log_A - log_Z);
        CMP_var(m,n) = exp(log_B - log_Z) - CMP_mean(m,n)^2;
        
    end
end

%% generate spikes
spk = zeros(size(lam));
rng(6)

for m = 1:size(lam, 1)
    for n = 1:size(lam, 2)
        spk(m,n) = com_rnd(lam(m,n), nu(m,n), 1);
        
    end
end

%% plot

subplot(1,3,1)
imagesc(dt:dt:T, x0,spk)
title('obs. spike counts')
ylabel('direction(rad)')
colorbar()
subplot(1,3,2)
imagesc(dt:dt:T, x0, CMP_mean)
title('Mean Firing Rate')
colorbar()
subplot(1,3,3)
imagesc(dt:dt:T, x0,CMP_var./CMP_mean)
title('Fano Factor')
xlabel('T')
colorbar()


