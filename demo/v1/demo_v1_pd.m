neuron=13;
% neuron=11;

nknots=7;
X = getCubicBSplineBasis(trial_x,nknots,true);
% X = [trial_x*0+1 cos(trial_x) sin(trial_x) cos(trial_x*2) sin(trial_x*2) cos(trial_x*3) sin(trial_x*3)];
x0 = linspace(0,2*pi,256)';
bas = getCubicBSplineBasis(x0,nknots,true);
% bas = [x0*0+1 cos(x0) sin(x0) cos(x0*2) sin(x0*2) cos(x0*3) sin(x0*3)];

% tblock = linspace(0,100,5);
tblock = linspace(0,120,5);
avgy=[]; b=[]; yhat=[]; dylo=[]; dyhi=[];
for j=1:(length(tblock)-1)
    tid = [1:size(trial_y,1)]>(tblock(j)*100) & [1:size(trial_y,1)]<(tblock(j+1)*100);
    for i=1:length(theta)
        avgy(i,j) = mean(trial_y(tid' & trial_x==theta(i),neuron));
    end
    
    [b(:,j),dev,stats] = glmfit(X(tid,:),trial_y(tid,neuron),'poisson','constant','off');
    [yhat(:,j),dylo(:,j),dyhi(:,j)]=glmval(b(:,j),bas,'log',stats,'constant','off');
end


%
figure(1)
clf
[~,sid]=sort(theta);
scatter(theta(sid)*180/pi,avgy(sid,1),'filled','MarkerFaceAlpha',0.5)
hold on
scatter(theta(sid)*180/pi,avgy(sid,end),'filled','MarkerFaceAlpha',0.5)
l1 = plot(x0*180/pi,yhat(:,1),'b','LineWidth',2);
l2 = plot(x0*180/pi,yhat(:,end),'r','LineWidth',2);
hold off
set(gca,'TickDir','out')
box off
xlabel('Grating Direction [deg]')
xlim([0 360])
ylabel('Spike Count')


cmap=lines(4);
[~,i]=max(yhat(:,1));
[~,theta_po] = min(abs(x0(i)-theta));
[~,j]=min(abs(x0-(theta(theta_po)-pi/2)));
hold on
plot(theta(theta_po)*180/pi,max(yhat(i,:))*1.01,'>','MarkerFaceColor',cmap(3,:),'MarkerSize',8)
plot(theta(theta_po)*180/pi-90,min(yhat(j,:))*.8,'^','MarkerFaceColor',cmap(4,:),'MarkerSize',8)
hold off
legend([l1 l2],{'early','late'})




%

%
figure(2)
clf
smoothing=15;
mff=[]; mmm=[]; mffse=[]; mmmse=[];
for i=1:(120-smoothing)
    ff=[];
    c=1;
    for stim=[theta(theta_po) theta(theta_po)-pi/2]
        tv=[];
        tv = find(abs(theta-stim)<20*pi/180);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):100:(100*smoothing)]+i*100];
        end
        ff = getFF(trial_y(tid,neuron),'bayes_bootstrap');
%         ff = getFF(trial_y(tid,neuron));
        mff(c,i) = mean(ff);
        mffse(c,i) = std(ff);
        mmm(c,i) = mean(trial_y(tid,neuron));
        mmmse(c,i) = std(trial_y(tid,neuron))/sqrt(length(tid));
        c=c+1;
    end
end
subplot(2,1,1)
plot([1:(120-smoothing)]+smoothing/2,mmm(1,:),'Color',cmap(3,:))
hold on
plot([1:(120-smoothing)]+smoothing/2,mmm(2,:),'Color',cmap(4,:))
plot([1:(120-smoothing)]+smoothing/2,mmm'+mmmse','k:')
plot([1:(120-smoothing)]+smoothing/2,mmm'-mmmse','k:')
hold off
xlabel('Trial')
ylabel('Mean')
box off; set(gca,'TickDir','out')
xlim([0 120])
subplot(2,1,2)
plot([1:(120-smoothing)]+smoothing/2,mff(1,:),'Color',cmap(3,:))
hold on
plot([1:(120-smoothing)]+smoothing/2,mff(2,:),'Color',cmap(4,:))
plot([1:(120-smoothing)]+smoothing/2,mff'+mffse','k:')
plot([1:(120-smoothing)]+smoothing/2,mff'-mffse','k:')
hold off
xlabel('Trial')
ylabel('Fano Factor')
box off; set(gca,'TickDir','out')
xlim([0 120])
% ylim([0.9 ])
line(xlim(),[1 1])