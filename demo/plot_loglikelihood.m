

y=10;

lv = linspace(0.001,20,64);
nv = linspace(0.5,5,50);
% nv = logspace(-1,2,50);
[lam,nu]=meshgrid(lv,nv);

logZ=lam*0;
for i=1:size(lam,1)
    for j=1:size(lam,2)
        c = logsum_calc(lam(i,j), nu(i,j), 1000);
        logZ(i,j) = c(1);
    end
end
ll=y.*log(lam)-nu.*gammaln(y+1)-logZ;
[i,j]=find(ll==max(ll(:)),1,'first');
% ll(ll<-100)=-100;

figure(1)
subplot(2,2,1)
% imagesc(lv,nv,log(ll-min(ll(:))))
imagesc(lv,nv,ll)
hold on
plot(lv(j),nv(i),'o') % mle in the range

% some lines
% for k=1:10
%     plot(lv,log(k)*log(lv),'k')
% end
c = nv(i)/log(lv(j));
plot(lv,c*log(lv),'k')

hold off
box off;; set(gca,'TickDir','out'); colorbar
set(gca,'YDir','normal')
subplot(2,2,2)
plot(nv,ll(:,j))

subplot(2,2,3)
plot(lv,ll(i,:))


lv2 = lv*1000
logZ_line=lv2*0;
for k=1:length(lv)
    tmp = logsum_calc(lv2(k), c*log(lv2(k)), 1000);
    logZ_line(k) = tmp(1);
end
ll_line=y.*log(lv2)-c*log(lv2).*gammaln(y+1)-logZ_line;
subplot(2,2,4)
plot(lv2,ll_line)
ylim([max(ll_line-0.1) max(ll_line)])



%% parameterized with mu

y=10;

nv = linspace(0,20,50);
mv = linspace(0.001,100,64);
[mu,nu]=meshgrid(mv,nv);
lam = mu.^nu;

logZ=lam*0;
for i=1:size(lam,1)
    for j=1:size(lam,2)
        c = logsum_calc(lam(i,j), nu(i,j), 1000);
        logZ(i,j) = c(1);
    end
end
ll=y.*log(lam)-nu.*gammaln(y+1)-logZ;
[i,j]=find(ll==max(ll(:)),1,'first');
% ll(ll<-100)=-100;

figure(1)
subplot(2,2,1)
% imagesc(lv,nv,log(ll-min(ll(:))))
imagesc(mv,nv,ll)
hold on
plot(mv(j),nv(i),'o')
hold off
box off;; set(gca,'TickDir','out'); colorbar
set(gca,'YDir','normal')
subplot(2,2,2)
plot(nv,ll(:,j))

subplot(2,2,3)
plot(mv,ll(i,:))

%% with beta and gamma

y=10;

bv = linspace(-1,2,64);
gv = linspace(-1,1,50);
% nv = logspace(-1,2,50);
[lam,nu]=meshgrid(exp(bv),exp(gv));

logZ=lam*0;
for i=1:size(lam,1)
    for j=1:size(lam,2)
        c = logsum_calc(lam(i,j), nu(i,j), 1000);
        logZ(i,j) = c(1);
    end
end
ll=y.*log(lam)-nu.*gammaln(y+1)-logZ;
[i,j]=find(ll==max(ll(:)),1,'first');
% ll(ll<-100)=-100;

figure(1)
clf
subplot(2,2,1)
% imagesc(lv,nv,log(ll-min(ll(:))))
imagesc(bv,gv,ll)
hold on
plot(bv(j),gv(i),'o') % mle in the range
hold off
box off;; set(gca,'TickDir','out'); colorbar
set(gca,'YDir','normal')



subplot(2,2,2)
plot(gv,ll(:,j))

subplot(2,2,3)
plot(bv,ll(i,:))

