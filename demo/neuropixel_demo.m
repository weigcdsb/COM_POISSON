addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

% usr_dir = 'C:\Users\gaw19004';
% r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
% r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];
%%
% dat_dir = 'D:\neuropixels\';
dat_dir = 'C:\Users\gaw19004\Documents\GitHub\data\pixel\';
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

figure(1)
subplot(1,2,1)
scatter(units.waveform_halfwidth+dw*(rand(n,1)-0.5),units.waveform_duration+yr,20,p(:,1),'filled')
set(gca,'TickDir','out')
ylabel('Duration')
xlabel('Half-Width')
subplot(1,2,2)
scatter(log10(units.firing_rate),units.waveform_duration+yr,20,p(:,1),'filled')
set(gca,'TickDir','out')
xlabel('Firing Rate')

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

figure(2)
clf
for i=1:25
    neuron = i+0;
    
    subplot(5,5,i)
    if p(neuron_id(neuron),1)>0.5
        plot(vvecd,S(neuron_id(neuron),:)+rand(1,size(S,2))-0.5,'k.')
    else
        plot(vvecd,S(neuron_id(neuron),:)+rand(1,size(S,2))-0.5,'b.')
    end
    segment_plot(vvecd(1:5000),S(neuron_id(neuron),1:5000),50)
    segment_plot(vvecd(5001:10000),S(neuron_id(neuron),5001:10000),50)
    segment_plot(vvecd(10001:15000),S(neuron_id(neuron),10001:15000),50)
    segment_plot(vvecd(15001:20000),S(neuron_id(neuron),15001:20000),50)
    segment_plot(vvecd(20001:25000),S(neuron_id(neuron),20001:25000),50)
    segment_plot(vvecd(25001:30000),S(neuron_id(neuron),25001:30000),50)
    segment_plot(vvecd(30001:35000),S(neuron_id(neuron),30001:35000),50)
%     segment_plot(vvecd(40001:50000),S(neuron_id(neuron),40001:50000),50)
%     segment_plot(vvecd(50001:60000),S(neuron_id(neuron),50001:60000),50)
%     segment_plot(vvecd(60001:70000),S(neuron_id(neuron),60001:70000),50)
%     segment_plot(vvecd(70001:80000),S(neuron_id(neuron),70001:80000),50)
end

%%

% examples --
% APN
%    6 unstable, under-disp
%    14/15 stable tuning, over and under
%    20 unstable, slight over-disp

% for ni=1:176
% neuron = id(ni);
neuron=neuron_id(6)
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

%%

segments = 20;
tsplit = linspace(0,T,segments+1);

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

% x1rgp = linspace(min(vvecd),max(vvecd),32);
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

%%

figure(7)
subplot(3,1,1)
plot(tvec(all(isfinite(X),2))/60,CMP_mean_fit)
ylim([0 25])
box off; set(gca,'TickDir','out')
for i=1:length(tsplit)
    line([1 1]*tsplit(i)/60,ylim())
end
xlim(xl);

subplot(3,1,2)
plot(tvec(all(isfinite(X),2))/60,CMP_ff_fit)
box off; set(gca,'TickDir','out')
xlim(xl);
subplot(3,1,3)
plot(tvec(all(isfinite(X),2))/60,CMP_ff_fit)
