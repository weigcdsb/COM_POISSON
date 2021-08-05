
function [lam,nu] = cmpDriftGenerator(seed,x,t,nutype)

% drifting von mises tuning

rng(seed)

% nu = repmat(linspace(0.5, 2, kStep), length(x0), 1);
if nargin<4,
    nutype='noisy_interp';
end
switch lower(nutype)
    case 'noisy_interp'
        tk = linspace(0,1,40);
        phs = interp1(tk,randn(size(tk)),t,'spline');
        phs = phs/2+pi;
        sclpts = randn(size(tk));
        scl = interp1(tk,sclpts,t,'spline');
        scl = scl/10 + 2;
        
        target_mean = 10;
        nuSing = interp1(tk,randn(size(tk)),t,'spline');
        nu = log(1+exp(nuSing))+0.5;
        offset=0.5;
    case 'overdisp'
        tk = linspace(0,1,10);
        phs = interp1(tk,randn(size(tk)),t,'spline');
        phs = phs*1.5+pi;
        sclpts = randn(size(tk));
        scl = interp1(tk,sclpts,t,'spline');
        scl = scl/20 + 1;
        
        target_mean = 20;
        nu = t*0+0.4;
        offset = 1;
end

lam = (scl.*cos(x-phs));
target_loglam = nu.*log(target_mean + (nu-1)/2./nu);
lam = log(1+exp(lam+target_loglam+offset));
