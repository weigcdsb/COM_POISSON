addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('C:\Users\ianhs\Documents\GitHub\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];
% usr_dir = 'C:\Users\ianhs';
% r_path = 'C:\Program Files\R\R-4.2.1\bin';
% r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%% load data

dat_dir = 'C:\Users\gaw19004\Documents\GitHub\data\pixel\';
% dat_dir = 'D:\neuropixels\';
rec_id = '719161530';

load([dat_dir rec_id '_spikes.mat'])
epoch = readtable([dat_dir rec_id '_epoch.csv']);
units = readtable([dat_dir rec_id '_units.csv']);
speed = readtable([dat_dir rec_id '_running_speed.csv']);

%% 

dt = 0.2;
T = max(cellfun(@max,Tlist));
S = getSpkMat(Tlist,dt,T,false);
tvec = linspace(dt/2,(ceil(T/dt)-dt/2)*dt,size(S,2));

%% get cell types

X = log([units.waveform_halfwidth units.waveform_duration units.firing_rate]);
GMModel = fitgmdist(X,2,'CovarianceType','diagonal');
p = GMModel.posterior(X);

if GMModel.mu(1,1)<GMModel.mu(2,1)
    p=fliplr(p);
end

n=size(units,1);
d = diff(sort(units.waveform_halfwidth));
d = d(d>10e-12); dw=d(1);
yr=dw*(rand(n,1)-0.5);


%% align running speed data with spikes

v = [(speed.start_time+speed.end_time)/2 speed.velocity];
v(v<-100)=NaN;
vvec = interp1(v(:,1),v(:,2),linspace(dt/20,T-dt/20,size(S,2)*10),'nearest');
vvec(~isfinite(vvec))=0;
vvecd = decimate(vvec,10);
vvecd(vvec(1:10:end)==0)=NaN;

%%

area = 'APN';
neuron_id = find(strcmp(units.ecephys_structure_acronym,area));

neuron=neuron_id(6);
% x1rg = linspace(min(vvecd),max(vvecd),64);
x1rg = linspace(0,max(vvecd),64);
x2rg = -0.5:(2*max(S(neuron,:))+0.5);
% [~,~,ns,~] = bindata2(vvecd*0,vvecd,full(S(neuron,:)),x1rg,x2rg);
[~,~,ns,~] = bindata2(vvecd*0,abs(vvecd),full(S(neuron,:)),x1rg,x2rg);

x1ax = x1rg(1:end-1)+mean(diff(x1rg))/2;
x2ax = 0:(2*max(S(neuron,:)));

figure(3)
clf
subplot(1,2,1)
imagesc(x1ax,x2ax,log10(ns'./sum(ns,2)'+0.01))
nsn=ns'./sum(ns,2)';
% imagesc(x1ax,x2ax,ns'./sum(ns,2)')
set(gca,'YDir','normal')
colorbar
title('Observed')
set(gca,'CLim',[-2 0])

% f = @(x) log(exp(x/2)+1);
fnl = @(x) (x+abs(x))/2;

y = full(S(neuron,:))';
% b = glmfit([vvecd'],y,'poisson');
% yhat = glmval(b,[x1ax'],'log');

hyp = @(x,p) abs(x)./(1+p*abs(x));
b = glmfit([hyp(vvecd',0.01)],y,'poisson');
yhat = glmval(b,[hyp(x1ax',0.01)],'log');

hold on
plot(x1ax,yhat)
hold off
subplot(1,2,2)
pdf=[];
for i=1:length(yhat)
    pdf(:,i) = poisspdf(x2ax,yhat(i));
end
imagesc(x1ax,x2ax,log10(pdf+0.01))
hold on
plot(x1ax,yhat)
hold off
title('Static Poisson')
set(gca,'YDir','normal')
set(gca,'CLim',[-2 0])
colorbar

figure(4)
clf
vest = x2ax.^2*nsn-(x2ax*nsn).^2;
mest = x2ax*nsn;
plot(x1ax,vest./mest,'o')
xlabel('Velocity')
ylabel('Fano Factor')

%% fit model

segments = 20;
tsplit = linspace(0,T,segments+1);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Qopt is not OK...
Qopt = 1e-5*ones(size(Qopt));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

%% plot

figure(5)
clf
subplot(4,segments,1:segments)
% bar(tvec/60,y,1)
stairs(tvec/60,y)
box off; set(gca,'TickDir','out')
axis tight
xl=xlim();
for i=1:length(tsplit)
    line([1 1]*tsplit(i)/60,ylim())
end

subplot(4,segments,(segments+1):2*segments)
plot(tvec/60,abs(vvecd))
axis tight; xlim(xl);
box off; set(gca,'TickDir','out')
for i=1:length(tsplit)
    line([1 1]*tsplit(i)/60,ylim())
end

clear cc_pdf cc_pdf_all
cc_pdf_all = zeros(length(x2ax),length(x1ax));
for i=1:segments
    tid = tvec>tsplit(i) & tvec<tsplit(i+1);
%     [~,~,ns,~] = bindata2(vvec(tid)*0,vvecd(tid),full(S(neuron,tid)),x1rg,x2rg);
    [~,~,ns,~] = bindata2(vvec(tid)*0,abs(vvecd(tid)),full(S(neuron,tid)),x1rg,x2rg);
    
    subplot(4,segments,2*segments+i)
    imagesc(x1ax,x2ax,log10(ns'./sum(ns,2)'+0.01))
    set(gca,'YDir','normal')
    set(gca,'CLim',[-2 0])
    set(gca,'TickDir','out'); box off
    if i>1
        set(gca,'YTickLabel',[])
        set(gca,'XTickLabel',[])
    end
    ylim([0 maxy/2])

    subplot(4,segments,3*segments+i)
    [~,bins]=histc(vvec(tid),x1rg);
    bins = bins(all(isfinite(X(tid,:)),2));
    ttid = find(tid(all(isfinite(X),2)));
    
%     zx = (x1ax-nanmean(X(:,2)))./nanstd(X(:,2));
    zx = hyp(x1ax,0.1);
    cc_pdf_f=full(ns')*0;
    cc_pdf=full(ns')*0;
    mg = nanmean(G(:,2));
    sg = nanstd(G(:,2));
    for j=1:length(ttid)
        lam = exp(theta_fit(1,ttid(j))+theta_fit(2,ttid(j))*zx);
%         lam = exp(theta_fit(1,ttid(j))+theta_fit(2,ttid(j))*zx+ theta_fit(3,ttid(j))*fx(-zx));
%         nu = exp(theta_fit(3,ttid(j)));
        nu = exp(theta_fit(3,ttid(j))+theta_fit(4,ttid(j))*(abs(x1ax)-mg)/sg);
        for k=1:length(lam)
%             cc_pdf(:,k) = com_pdf(0:maxy,lam(k),exp(theta_fit(end,ttid(j))));
            cc_pdf(:,k) = com_pdf(x2ax,lam(k),nu(k));
        end
        cc_pdf_f = cc_pdf_f+cc_pdf;
        cc_pdf_all = cc_pdf_all+cc_pdf;
    end
    imagesc(x1ax,x2ax,log10(cc_pdf_f/length(ttid)+0.01))
    hold on
    plot(x1ax,x2ax*(cc_pdf_f./sum(cc_pdf_f)))
    hold off
    m1(i,:) = x2ax*(cc_pdf_f./sum(cc_pdf_f));
    m2(i,:) = x2ax.^2*(cc_pdf_f./sum(cc_pdf_f));
%     
%     cc = cmp_pdf(:,tid);
% %     cmp_pdf(:,tid)
% %     keyboard
%     ns_mod=ns*0;
%     for j=1:max(bins)
%         if sum(bins==j)>0
%             ns_mod(j,:) = (sum(cc(:,bins==j),2)/sum(bins==j))';
%         end
%     end
%     imagesc(x1ax,x2ax,log10(ns_mod'+0.01))
    set(gca,'YDir','normal')
    set(gca,'CLim',[-2 0])
    set(gca,'TickDir','out'); box off
    if i>1
        set(gca,'YTickLabel',[])
        set(gca,'XTickLabel',[])
    end
    ylim([0 maxy/2])
    drawnow
%   
end

%%
x1rgp = linspace(0,max(vvecd),16);
x1axp = x1rgp(1:end-1)+mean(diff(x1rgp))/2;

cmap = hot(segments-1);
figure(6)
clf
id = all(isfinite(X),2);
Xid = X(id,:);
for i=3:3:(segments-1)
    tid = tvec>tsplit(i) & tvec<tsplit(i+1);
%     [~,~,ns,~] = bindata2(vvec(tid)*0,vvecd(tid),full(S(neuron,tid)),x1rgp,x2rg);
    [~,~,ns,~] = bindata2(vvec(tid)*0,abs(vvecd(tid)),full(S(neuron,tid)),x1rgp,x2rg);
    nsn = ns'./sum(ns,2)';
    vest = x2ax.^2*nsn-(x2ax*nsn).^2;
    mest = x2ax*nsn;
        
    tid = tvec(id)>tsplit(i) & tvec(id)<tsplit(i+1);
    
%     [~,~,ns,~] = bindata2(Xid(tid,2)'*0,Xid(tid,2)',CMP_mean_fit(tid),x1rgp,x2rg);
%     nsn = ns'./sum(ns,2)';
%     v1est = x2ax.^2*nsn-(x2ax*nsn).^2;
%     m1est = x2ax*nsn;
% 
%     [~,~,ns,~] = bindata2(Xid(tid,2)'*0,Xid(tid,2)',CMP_var_fit(tid),x1rgp,x2rg);
%     nsn = ns'./sum(ns,2)';
%     v2est = x2ax.^2*nsn-(x2ax*nsn).^2;
%     m2est = x2ax*nsn;
% 
%     totvar = v1est+m2est;

    subplot(2,2,1)
    scatter(x1axp,mest,10,cmap(i,:),'filled')
    hold on
    plot(x1ax,m1(i,:),'Color',cmap(i,:));

    subplot(2,2,3)
    scatter(x1axp,vest./mest,10,cmap(i,:),'filled')
    hold on
    plot(x1ax,(m2(i,:)-m1(i,:).^2)./m1(i,:),'Color',cmap(i,:));
end

% Bayesian bootstrap...
[~,bins]=histc(vvecd',x1rgp);
mm=zeros(1000,max(bins)-1);
ff=zeros(1000,max(bins)-1);
for i=1:(max(bins)-1)
    t = full(S(neuron,bins==i))';
    theta = rdirichlet(1000,repmat(1,1,length(t)));
    wm = t'*theta;
    wv = sum(theta.*(bsxfun(@minus,t,wm).^2));
    ff(:,i) = wv./wm;
    mm(:,i) = wm;
end
mmq = prctile(mm,[5 50 95]);
ffq = prctile(ff,[5 50 95]);

subplot(2,2,1)
title(num2str(neuron))
hold off
box off; set(gca,'TickDir','out'); ylabel('Mean')
axis tight
subplot(2,2,3)
hold off
box off; set(gca,'TickDir','out'); ylabel('Fano Factor')
xlabel('Velocity')
axis tight


subplot(2,2,2)
errorbar(x1axp,mmq(2,:),mmq(2,:)-mmq(1,:),mmq(3,:)-mmq(2,:),'.','CapSize',0)
hold on
plot(x1ax,x2ax*(cc_pdf_all./sum(cc_pdf_all)))
hold off
axis tight
set(gca,'TickDir','out'); box off


subplot(2,2,4)
errorbar(x1axp,ffq(2,:),ffq(2,:)-ffq(1,:),ffq(3,:)-ffq(2,:),'.','CapSize',0)
hold on
xlabel('Velocity')
cn = cc_pdf_all./sum(cc_pdf_all);
plot(x1ax,(x2ax.^2*cn-(x2ax*cn).^2)./(x2ax*cn))
hold off
axis tight
set(gca,'TickDir','out'); box off

subplot(2,2,1)
ylim([0 15])
% ylim([5 20])
subplot(2,2,2)
ylim([0 15])
% ylim([5 20])
subplot(2,2,3)
ylim([0 3])
% ylim([0 0.7])
subplot(2,2,4)
ylim([0 3])
% ylim([0 0.7])

% waitforbuttonpress()
% end




