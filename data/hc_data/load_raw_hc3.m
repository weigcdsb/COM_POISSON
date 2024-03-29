
addpath('crcns-hc2-scripts')

% prefix='C:\Users\ian\Documents\MATLAB\data_spk\crcns_hc3\ec013.40\';
% base_name = 'ec013.718'; % linear track - 62

% prefix='C:\Users\ian\Documents\MATLAB\data_spk\crcns_hc3\ec014.29\';
% base_name = 'ec014.468'; % linear - 97

% prefix='C:\Users\ian\Documents\MATLAB\data_spk\crcns_hc3\ec016.19\';
% base_name = 'ec016.269'; % linear track - 90

prefix='C:\Users\ian\Documents\MATLAB\data_spk\crcns_hc3\gor01-6-7\';
base_name = '2006-6-7_11-26-53'; % linearOne - 86

%

% Load Spikes
[T,G,Map,Par]=LoadCluRes([prefix base_name '\' base_name]);

% Load head x-y, body x-y
pos_raw=dlmread([prefix base_name '\' base_name '.whl']);

% Load LFP
% lfp = LoadBinary([prefix base_name '\' base_name '.eeg'],1:10);

%% Get spike waveforms...
% 
% waveforms = cell(0);
% gwave = cell(0);
% for i=1:length(Par.SpkGrps)
%     waveforms{i} = LoadSpk([prefix base_name '\' base_name sprintf('.spk.%i',i)],length(Par.SpkGrps(i).Channels),Par.SpkGrps(i).nSamples);
%     gwave{i} = LoadClu([prefix base_name '\' base_name sprintf('.clu.%i',i)]);
% end
% 
% 
% %% Plot waveforms...
% figure(10); clf
% spkgrp = 1;
% ug = unique(gwave{spkgrp});
% cmap = hsv(length(ug));
% grpMap = Map(Map(:,2)==spkgrp,:);
% for i=1:length(ug)
%     idx = find(gwave{spkgrp}==ug(i));
%     wlo = min(min(min(waveforms{spkgrp}(:,:,idx))));
%     whi = max(max(max(waveforms{spkgrp}(:,:,idx))));
%     for j=1:size(waveforms{spkgrp},1)
%         subplot(size(waveforms{spkgrp},1),length(ug),(j-1)*length(ug)+i)
%         plot(squeeze(waveforms{spkgrp}(j,:,idx(1:min(200,length(idx))))),'Color',cmap(i,:))
%         xlim([1 size(waveforms{spkgrp},2)])
%         ylim([wlo whi])
%         set(gca,'XTick',[]);
%         set(gca,'YTick',[]);
%         if j==1
%             tmp = (grpMap(:,3)==ug(i));
%             if any(tmp)
%                 title(sprintf('Neuron %i',grpMap(tmp,1)))
%             end
%         end
%         if i==1
%             ylabel(sprintf('e%02i',Par.SpkGrps(spkgrp).Channels(j)))
%         end
%         box off
%     end
% end

%% 
% 
% Fs=1250; % Original sampling frequency of LFP
% Ds=64;  % Downsample rate
% [b,a]=butter(4,[8 12]*2/Fs,'bandpass'); %Band-pass filter
% 
% lfp_filt=[];
% for i=1:size(lfp,1)
%     lfp_filt(i,:) = resample(filtfilt(b,a,lfp(i,:)),Ds,Fs);
% end
% theta = hilbert(lfp_filt(1,:));
% ph = angle(theta);

%% Get position and spike counts at sampling rate Ds units (Hz)

% Resample position...
Ds = 5; % sampling rate in Hz
Par.VideoSampleRate = 39.06; % original video sampling rate
pos = pos_raw;
pos(pos==-1)=NaN;
poshf=[];
t_in  = linspace(0,size(pos,1)/39.06,size(pos,1));
t_out = linspace(0,size(pos,1)/39.06,floor(size(pos,1)/39.06*Ds));

for i=1:size(pos,2)
    poshf(:,i) = interp1(t_in,pos(:,i),t_out,'nearest');
    tn = 1:length(poshf(:,i));
    nanid = isnan(poshf(:,i));
    poshf(nanid,i)= interp1(tn(~nanid),poshf(~nanid,i),tn(nanid),'nearest','extrap');
end
pos=poshf;

% Bin spikes...
uG = unique(G);
spk=zeros(size(poshf,1),length(uG));
for i=1:length(uG)
    idx=find(G==uG(i));
    spk(:,i)=hist(T(idx)/Par.SampleRate,linspace(0,size(poshf,1)/Ds,size(poshf,1)));
end
% 
% lo_samp_rate = 1;
% spk_lo=[];
% for i=1:length(uG)
%     idx=find(G==uG(i));
%     spk_lo(:,i)=hist(T(idx)/Par.SampleRate,linspace(0,size(poshf,1)/Ds,round((size(poshf,1)/Ds*lo_samp_rate))));
% end


% nanInds = any(isnan(pos)');
% pos = pos(~nanInds,:);
% spk = spk(~nanInds,:);

% if length(nanInds)>length(ph)
%     ph  = ph(~nanInds(1:length(ph)));
% else
%     ph = ph(1:length(nanInds));
%     ph = ph(~nanInds);
% end

%% reorient track to get one-D data
rpos = bsxfun(@minus,pos,mean(pos));
[rpos,u,v] = svd(rpos(:,1:2),'econ');
rrpos=rpos*u';
rpos(:,1) = rpos(:,1)-min(rpos(:,1));
rpos(:,2)=rpos(:,2)/(max(rpos(:,1))+.00001);
rpos(:,1) = rpos(:,1)/(max(rpos(:,1))+.00001);

bounds = [.1 .9];           %Part of track to consider runway
b = nan*ones(size(rpos,1),1);
b(rpos(:,1) < bounds(1)) = -1;
b(rpos(:,1) > bounds(2)) = 1;

nanInds = find(~isnan(b));
b = interp1(nanInds,b(nanInds),1:size(pos,1));
b = [0 diff(b)];
runs = bwlabel(b > 0);

w = watershed(b==0);
w = w-1; %w(w== max(w)) = 0;

figure(1)
clf
subplot(2,5,1)
plot(pos(:,1),pos(:,2))
title('original')
axis equal
subplot(2,5,6)
plot(rrpos(:,1),rrpos(:,2))
title('re-oriented')
axis equal
subplot(2,5,[2:5 7:10])
% separate the runs into two directions...
t = 1:length(w);
for i=1:max(w)
    if mod(i,2)==0
        plot(t(w==i),rpos(w==i,1),'b')
    else
        plot(t(w==i),rpos(w==i,1),'r')
    end
    hold on
end
hold off
title('segmented')

% get "circular" position [0 2] to separate diff running directions
cpos = rpos(:,1);
cpos(mod(w,2)==0) = 2-rpos(mod(w,2)==0,1);

%%

% relabel for clarity...
position = pos;
position_realigned = rpos;
position_circular = cpos;
spike_counts = spk;
run_number = w;

save([strrep(base_name,'.','_') '_preprocessed_5Hz'],'base_name','position','position_realigned','position_circular','spike_counts','run_number')

%% Plot 2d dot maps
figure(2)
clf
for i=1:size(spk,2)
    subplot(10,8,i)
    if any(spk(:,i))
        plot(pos(:,1),pos(:,2),'b.')
        hold on
        plot(pos(spk(:,i)>0,1),pos(spk(:,i)>0,2),'ro')
        hold off
        xlim([min(pos(:,1)) max(pos(:,1))])
        ylim([min(pos(:,2)) max(pos(:,2))])
        axis equal
        set(gca,'XTick',[]);
        set(gca,'YTick',[]);
        drawnow
    end
end

%% Plot 1d dot maps

figure(3); clf
t = 1:size(rpos,1); c=1;
% for neuron=1:size(spk,2)
for neuron=1:size(spk,2)
    subplot(3,20,c)
    plot(rpos(:,1),t,'k')
    hold on
    plot(rpos(spk(:,neuron)>0,1),t(spk(:,neuron)>0),'r.')
    hold off
%     plot(cpos(:,1),t,'k')
%     hold on
%     plot(cpos(spk(:,neuron)>0,1),t(spk(:,neuron)>0),'r.')
%     hold off
    set(gca,'XTick',[]);
    set(gca,'YTick',[]);
    drawnow
    axis tight
    c=c+1;
end

% %% Plot phase locking...
% 
% figure(4); clf
% for neuron=1:size(spk,2)
%     subplot(4,20,neuron)
%     hist(ph(spk(:,neuron)>0),100)
%     xlim([-pi pi])
%     set(gca,'XTick',[]);
%     set(gca,'YTick',[]);
%     drawnow
% end

% %% Plot phase precession...
% 
% figure(5); clf
% % for neuron=1:size(spk,2)
% %     subplot(4,4,neuron)
% for neuron=32
%     plot(cpos(spk(:,neuron)>0,1),ph(spk(:,neuron)>0),'k.')
%     xlim([0 2])
%     ylim([-pi pi])
%     drawnow
%     ylabel('Theta Phase')
%     xlabel('Position')
% end

% 
% 
% x0 = linspace(0,1,128);
% [n,binx] = histc(rpos(:,1),x0);
% trial_resp=zeros(max(w),length(x0),size(spk,2));
% for neuron=20
% for i=1:max(w)
%     for j=1:length(x0)
%         idx = (w==i) & (binx==j)';
%         if any(idx)
%             trial_resp(i,j,neuron) = sum(spk(idx,neuron))/length(idx);
%         end
%     end
% end
% end
% 
% %%
% figure(1)
% clf
% colormap hot
% % for neuron=1:size(spk,2)
% for neuron=20
% %     subplot(10,8,neuron)
%     imagesc(trial_resp(:,:,neuron))
%     imagesc([trial_resp(1:2:end-1,:,neuron) fliplr(trial_resp(2:2:end,:,neuron))])
%     drawnow
%     set(gca,'XTick',[]);
% 	set(gca,'YTick',[]);
% end