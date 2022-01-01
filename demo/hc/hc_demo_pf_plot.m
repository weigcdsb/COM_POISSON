
% load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion

idx0 = 1;
pos_raw = (position_circular(idx0:end)-1)*cam2cm;
posAlign_raw = position_realigned(idx0:end,1)*cam2cm;
t_raw = linspace(0,size(pos_raw,1),size(pos_raw,1))/5/60;
t_raw = t_raw+mean(diff(t_raw))/2;

% no coarse bin
bin = 1;
t = t_raw;
pos = pos_raw;

neuron=12;
place_field_range = [-110 -60];

% neuron=35
% place_field_range = [60 140];

spk_raw = spike_counts(idx0:end,neuron);

figure(21)
subplot(3,1,1)

% plot(t_raw,pos)
% hold on
% scatter(t_raw(find(spk_raw>0)),pos(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
% hold off
% ylim([-250 250])
% line(xlim(),[1 1]*place_field_range(1),'Color','r')
% line(xlim(),[1 1]*place_field_range(2),'Color','r')

plot(t_raw,posAlign_raw)
hold on
scatter(t_raw(find(spk_raw>0)),posAlign_raw(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
hold off
ylim([0 max(posAlign_raw)])
line(xlim(),[1 1]*(250-place_field_range(1)),'Color','r')
line(xlim(),[1 1]*(250-place_field_range(2)),'Color','r')
line(xlim(),[1 1]*(250+place_field_range(1)),'Color','r')
line(xlim(),[1 1]*(250+place_field_range(2)),'Color','r')

box off; set(gca,'TickDir','out')
xlim([0 t_raw(end)])
ylabel('Position [cm]')

subplot(3,1,2)
id_at_pf = (pos>place_field_range(1) & pos<place_field_range(2));
plot(t_raw(id_at_pf),spk_raw(id_at_pf),'o')
xlim([0 t_raw(end)])
box off

% get moving average mean and ff
bin_centers = linspace(0,floor(t_raw(end)*10)/10,floor(t_raw(end))*10); % 1min stride
bin_width = 10;
mpf=[]; ffpf=[]; ffpfse=[];
for i=1:length(bin_centers)
    id_at_pf_bin = (id_at_pf' & t_raw>(bin_centers(i)-bin_width/2) & t_raw<(bin_centers(i)+bin_width/2));
    if ~isempty(id_at_pf_bin)
        mpf(i) = mean(spk_raw(id_at_pf_bin));
        ffpf(i) = var(spk_raw(id_at_pf_bin))./mean(spk_raw(id_at_pf_bin));
%         
        ff = getFF(spk_raw(id_at_pf_bin),'bayes_bootstrap');
        ffpf(i) = mean(ff);
        ffpfse(i) = std(ff);
    else
        mpf(i) = NaN;
        ffpf(i) = NaN;
    end
end
subplot(3,1,2)
hold on
plot(bin_centers,mpf);
hold off
set(gca,'TickDir','out')
ylabel('Spike Count')

subplot(3,1,3)
plot(bin_centers,ffpf);
hold on
plot(bin_centers,ffpf+ffpfse,'k:')
plot(bin_centers,ffpf-ffpfse,'k:')
hold off
box off
set(gca,'TickDir','out')
xlim([0 t_raw(end)])
ylabel('Fano Factor')
line(xlim(),[1 1])

% 
% %%
% 
% subplot(3,1,1)
% plot(t_raw,pos)
% hold on
% scatter(t_raw(find(spk_raw>0)),pos(spk_raw>0),spk_raw(spk_raw>0)*10,'filled','k','MarkerFaceAlpha',0.5)
% hold off
% box off; set(gca,'TickDir','out')
% ylim([-250 250])
% xlim([0 t_raw(end)])
% subplot(3,1,2)
% cla
% subplot(3,1,3)
% cla
% 
% mpf=[]; ffpf=[];
% pf_center = linspace(-200,200,20);
% cmap = hsv(length(pf_center));
% for j=1:length(pf_center)
%     place_field_range=[pf_center(j)-20 pf_center(j)+20];
%     id_at_pf = (pos>place_field_range(1) & pos<place_field_range(2));
%     bin_centers = linspace(0,floor(t_raw(end)*10)/10,floor(t_raw(end))*10); % 1min stride
%     bin_width = 10;
%     
%     for i=1:length(bin_centers)
%         id_at_pf_bin = (id_at_pf' & t_raw>(bin_centers(i)-bin_width/2) & t_raw<(bin_centers(i)+bin_width/2));
%         if ~isempty(id_at_pf_bin)
%             mpf(i,j) = mean(spk_raw(id_at_pf_bin));
%             ffpf(i,j) = var(spk_raw(id_at_pf_bin))./mean(spk_raw(id_at_pf_bin));
%         else
%             mpf(i,j) = NaN;
%             ffpf(i,j) = NaN;
%         end
%     end
%     subplot(3,1,2)
%     hold on
%     plot(bin_centers,mpf(:,j),'Color',cmap(j,:));
%     hold off
%     subplot(3,1,3)
%     hold on
%     plot(bin_centers,ffpf(:,j),'Color',cmap(j,:));
%     hold off
% end
% subplot(3,1,2)
% set(gca,'TickDir','out')
% xlim([0 t_raw(end)])
% box off
% subplot(3,1,3)
% set(gca,'TickDir','out')
% xlim([0 t_raw(end)])
% box off
