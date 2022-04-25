addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

%%
load('ec014_468_preprocessed_5Hz.mat')
cam2cm = 250/range(prctile(position_realigned(:,1),[1 99])); % unit conversion
nNeuron = size(spike_counts, 2);

idx0 = 1;
pos = (position_circular(idx0:end)-1)*cam2cm;

TESTID = [];
QOPT = zeros(nNeuron, 2);
LLHD_train = zeros(nNeuron, 5)*NaN;
LLHD_test = zeros(nNeuron, 5)*NaN;
LLHD_spk_train = zeros(nNeuron, 5)*NaN;
LLHD_spk_test = zeros(nNeuron, 5)*NaN;
BIT_spk_train = zeros(nNeuron, 4)*NaN;
BIT_spk_test = zeros(nNeuron, 4)*NaN;


rng(1)
for neuron = 1:nNeuron
    disp("neuron"+ neuron);
    c = 0;
    flg = 1;
    runInit = 2;
    while(flg)
        try
            [~,TESTID{neuron}, QOPT(neuron,:), LLHD_train(neuron,:),...
                LLHD_test(neuron,:), llhd_spk, bit_spk] =...
            evac('model_run(neuron,runInit,run_number,pos,spike_counts,1/20,12,usr_dir,r_path,r_wd)');
            LLHD_spk_train(neuron,:) = llhd_spk(1,:);
            LLHD_spk_test(neuron,:) = llhd_spk(2,:);
            BIT_spk_train(neuron,:) = bit_spk(1,:);
            BIT_spk_test(neuron,:) = bit_spk(2,:);
            flg = 0;
            disp(BIT_spk_test(neuron,:))
            
        catch
            disp('rep')
            runInit = runInit + 1;
            c = c + 1;
        end
        
        if c>=3
            flg = 0;
        end
    end
end


reNeuronID = [];
for kk = 1:nNeuron
    mTmp = min(BIT_spk_train(kk,:));
    disp(mTmp)
    if (isnan(mTmp) || mTmp < 0 || abs(mTmp)> 10)
        reNeuronID = [reNeuronID kk];
    end
end

%%
useID = setdiff(1:nNeuron, reNeuronID);
length(useID)

BIT_spk_train_use = BIT_spk_train(useID,:);
BIT_spk_test_use = BIT_spk_test(useID,:);


cd('C:\Users\gaw19004\Documents\GitHub\COM_POISSON\plots\figure5')
mod_compare = figure;
subplot(2,1,1)
hold on
plot(BIT_spk_train_use','Color', [0.5, 0.5, 0.5, 0.3])
q1 = quantile(BIT_spk_train_use,0.25);
medVal = median(BIT_spk_train_use);
q3 = quantile(BIT_spk_train_use,0.75);
plot(medVal,'r', 'LineWidth',2)
plot(q1, 'r--', 'LineWidth',2)
plot(q3, 'r--', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.25,medVal(t)+1.5,num2str(round(medVal(t),3)),'Color','red','FontSize',10)
end
hold off
xlim([0.5 4.5])
xticks(1:4)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
title('bit/spk-train')

subplot(2,1,2)
hold on
plot(BIT_spk_test_use','Color', [0.5, 0.5, 0.5, 0.3])
q1 = quantile(BIT_spk_test_use,0.25);
medVal = median(BIT_spk_test_use);
q3 = quantile(BIT_spk_test_use,0.75);
plot(medVal,'r', 'LineWidth',2)
plot(q1, 'r--', 'LineWidth',2)
plot(q3, 'r--', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.25,medVal(t)+1.5,num2str(round(medVal(t),3)),'Color','red','FontSize',10)
end
hold off
xlim([0.5 4.5])
xticks(1:4)
xticklabels({'dCMP-(12,1)','dPoi-(12)','sCMP-(12,1)','sPoi-(12)'})
xtickangle(45)
set(gca,'FontSize',10, 'LineWidth', 1.5,'TickDir','out')
box off
title('bit/spk-test')

set(mod_compare,'PaperUnits','inches','PaperPosition',[0 0 4 4])
saveas(mod_compare, '5_modComp.svg')
saveas(mod_compare, '5_modComp.png')













%%
% for neuron = reNeuronID
%     disp("neuron"+ neuron);
%     c = 0;
%     flg = 1;
%     runInit = 1;
%     while(flg)
%         try
%             [~,TESTID{neuron}, QOPT(neuron,:), LLHD_train(neuron,:),...
%                 LLHD_test(neuron,:), llhd_spk, bit_spk] =...
%             evalc('model_run(neuron,runInit,run_number,pos,spike_counts,1/20,12,usr_dir,r_path,r_wd)');
%             LLHD_spk_train(neuron,:) = llhd_spk(1,:);
%             LLHD_spk_test(neuron,:) = llhd_spk(2,:);
%             BIT_spk_train(neuron,:) = bit_spk(1,:);
%             BIT_spk_test(neuron,:) = bit_spk(2,:);
%             flg = 0;
%             disp(BIT_spk_test(neuron,:))
%             
%         catch
%             disp('rep')
%             runInit = runInit + 1;
%             c = c + 1;
%         end
%         
%         if c>=3
%             flg = 0;
%         end
%     end
% end











