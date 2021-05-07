addpath(genpath('D:\GitHub\COM_POISSON'));
%
% %%
load('data_monkey1_gratings_movie.mat')
load('gratings_movie.mat')
load('theta_gratings_movie.mat')
% see http://crcns.org/data-sets/vc/pvc-11/about

%% Show stimulus

% h = imagesc(M(:,:,1));
% axis equal off
% colormap gray
% for i=1:size(M,3)
%     set(h,'CData',M(:,:,i))
%     drawnow
%     pause(1/30)
% end

%%
neuron=2;
Tlist=cell(0);
c=1;
for i=1:size(data.EVENTS,2)
    Tlist{c} = data.EVENTS{neuron,i};
    c=c+1;
end

figure(2)
clf
drawRaster(Tlist,1,0,[0 30])

%% Get counts per trial

trial_x=repmat(theta',size(data.EVENTS,2),1);

trial_y=[];
c=1;
stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
            % Q1: t + 0.05??
        end
        t=t+stim_length;
        c=c+1;
    end
end


%%
neuron=24;

for i=1:length(theta)
    s(i,2)=var(trial_y(i:100:6000,neuron));
    s(i,1)=mean(trial_y(i:100:6000,neuron));
end

figure(2)
subplot(1,3,1)
% s(:, 1) = mean for 100 theta
% s(:, 2) = var for 100 theta
% plot var vs. mean
plot(s(:,1),s(:,2),'.','MarkerSize',20)

for i=1:length(theta)
    s(i,2)=var(trial_y([i:100:6000]+6000,neuron));
    s(i,1)=mean(trial_y([i:100:6000]+6000,neuron));
end
hold on
plot(s(:,1),s(:,2),'.','MarkerSize',10)
hold off
line(xlim,xlim)
xlabel('Mean')
ylabel('Variance')
legend({'First Half','Second Half'})

%

smoothing=20;
mff=[]; mmm=[];
for i=1:(120-smoothing)
    ff=[];
    c=1;
    for stim=linspace(0,2*pi,10)
        tv=[];
        tv = find(abs(theta-stim)<pi/5);
        tid=[];
        for s=1:length(tv)
            tid = [tid [tv(s):100:(100*smoothing)]+i*100];
        end
%         ff = getFF(trial_y(tid,neuron),'np_bootstrap');
        ff = getFF(trial_y(tid,neuron));
        mff(c,i) = mean(ff);
        mmm(c,i) = mean(trial_y(tid,neuron));
        c=c+1;
    end
end
subplot(1,3,2)
plot([1:(120-smoothing)]+smoothing/2,mmm')
xlabel('Trial')
ylabel('Mean')
subplot(1,3,3)
plot([1:(120-smoothing)]+smoothing/2,mff')
xlabel('Trial')
ylabel('Fano Factor')

%%


figure(11)
neuron=13;
[~,idx]=sort(theta);
ry = reshape(trial_y(:,neuron),100,[]);


% compare
subplot(1, 2, 1)
hold on
for k = 1:120
    plot(theta', ry(:, k),'.')
end
hold off
subplot(1, 2, 2)
plot(trial_x,trial_y(:,neuron),'.')


figure(12)
subplot(1,2,1)
plot(trial_x,trial_y(:,neuron),'.')
xlabel('Stimulus Direction [rad]')
subplot(1,2,2)
imagesc(ry(idx,:))
xlabel('Trials')
ylabel('Stimulus Number')

% Q2:
% ylabel = 'stimulus number'
% ylabel = 'theta/ direction'




%% Example tuning curve fit
nknots=7;
X = getCubicBSplineBasis(trial_x,nknots,true);
x0 = linspace(0,2*pi,256);
bas = getCubicBSplineBasis(x0,nknots,true);

[b,dev,stats] = glmfit(X,trial_y(:,neuron),'poisson','constant','off');
[yhat,dylo,dyhi]=glmval(b,bas,'log',stats,'constant','off');


figure(12)
plot(trial_x,trial_y(:,neuron),'.')
hold on
plot(x0,yhat)
plot(x0,yhat-dylo)
plot(x0,yhat+dyhi)
hold off