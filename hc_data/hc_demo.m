
% Raw datasets...
% https://portal.nersc.gov/project/crcns/download/hc-3/ec013.40/ec013.718.tar.gz
% https://portal.nersc.gov/project/crcns/download/hc-3/ec014.29/ec014.468.tar.gz
% https://portal.nersc.gov/project/crcns/download/hc-3/ec016.19/ec016.269.tar.gz

% Processed datasets...
% load('ec013_718_preprocessed_5Hz.mat')
load('ec014_468_preprocessed_5Hz.mat')
% load('ec016_269_preprocessed_5Hz.mat')


%%

% linear track is 250cm (http://crcns.org/files/data/hc3/crcns-hc3-data-description.pdf)
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion

%% Plot position with spikes overlaid as dots

figure(3); clf
t = linspace(0,size(position,1),size(position,1))/5/60;
t = t+mean(diff(t))/2;

for neuron=45
    subplot(1,3,1)
    plot(t,position_realigned(:,1)*cam2cm,'k')
    hold on
    plot(t(spike_counts(:,neuron)>0),position_realigned(spike_counts(:,neuron)>0,1)*cam2cm,'r.')
    hold off

    xlabel('Time [min]')
    ylabel('Position [cm]')
    box off; set(gca,'TickDir','out')
    
    subplot(1,3,2)
    plot((position_circular-1)*cam2cm,spike_counts(:,neuron),'.')
    xlabel('Position [cm]')
    ylabel('Spike Counts')
    box off; set(gca,'TickDir','out')
    
    subplot(1,3,3)
    plot(spike_counts(:,neuron))
end

%% Plot place fields (raw)

x0 = linspace(0,2,128);
[n,binx] = histc(position_circular(:,1),x0);
lam=[];
for i=1:length(x0)
    lam(:,i) = mean(spike_counts(binx==i,:));
end
[tmp,midx]=max(lam');
[tmp,bidx]=sort(midx);

figure(6); clf
cx0=x0+mean(diff(x0))/2-1;

subplot(1,2,1)
imagesc(cx0*cam2cm,1:size(lam,1),lam)
box off; set(gca,'TickDir','out')
xlabel('Position [cm]')
ylabel('Neuron')

subplot(1,2,2)
imagesc(cx0*cam2cm,1:size(lam,1),lam(bidx,:))
box off; set(gca,'TickDir','out')
xlabel('Position [cm]')
ylabel('Neuron')
