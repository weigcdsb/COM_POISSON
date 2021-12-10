addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
%%
load('data_monkey1_gratings_movie.mat')
load('theta_gratings_movie.mat')

%% samples

% full
trial_x_full=repmat(theta',size(data.EVENTS,2),1);
trial_y_full=[];
c=1;
stim_length=0.3;
for rep=1:size(data.EVENTS,2)
    t=0;
    for i=1:length(theta)
        for neuron=1:size(data.EVENTS,1)
            trial_y_full(c,neuron) = sum(data.EVENTS{neuron,rep}>(t+0.05) & data.EVENTS{neuron,rep}<(t+stim_length));
        end
        t=t+stim_length;
        c=c+1;
    end
end

% sub
rng(1)

nAll = length(theta);
nSS = round(nAll/2);

LLHD = [];
BIT = [];
SSIDX = [];

for n = 1:size(trial_y_full,2)
    flag = true;
    while flag
        ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
        for k = 1:size(data.EVENTS, 2)
            ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
                sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
        end
        SSIDX{n} = ssIdx;
        try
            [LLHD{n}, BIT{n}] = models_run(n, trial_x_full, trial_y_full,...
                ssIdx, nSS, nAll, false, usr_dir, r_path);
            flag = false;
        catch
            disp('error: resample...')
        end
    end
end


%% plot
llhd_mat = cell2mat(LLHD');
llhd_train = llhd_mat(1:2:end,:);
llhd_test = llhd_mat(2:2:end,:);


% plot(llhd_train(setdiff(1:size(trial_y_full,2), nPoiFail),:)')
% plot(llhd_test(setdiff(1:size(trial_y_full,2), nPoiFail),:)')

bit_mat = cell2mat(BIT');
bit_train = bit_mat(1:2:end,:);
bit_test = bit_mat(2:2:end,:); 
    

figure(1)
subplot(2,1,1)
hold on
plot(bit_train','Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_train);
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('train')

subplot(2,1,2)
hold on
plot(bit_test','Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_test);
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('test')



nPoiFail = find(llhd_train(:,3) < -10);
figure(2)
subplot(2,1,1)
hold on
plot(bit_train(setdiff(1:size(trial_y_full,2), nPoiFail),:)',...
    'Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_train(setdiff(1:size(trial_y_full,2), nPoiFail),:));
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('train')

subplot(2,1,2)
hold on
plot(bit_test(setdiff(1:size(trial_y_full,2), nPoiFail),:)',...
    'Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_test(setdiff(1:size(trial_y_full,2), nPoiFail),:));
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('test')



nPoiFail2 = find(llhd_train(:,3) < llhd_train(:,5));

figure(3)
subplot(2,1,1)
hold on
plot(bit_train(setdiff(1:size(trial_y_full,2), nPoiFail2),:)',...
    'Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_train(setdiff(1:size(trial_y_full,2), nPoiFail2),:));
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('train')

subplot(2,1,2)
hold on
plot(bit_test(setdiff(1:size(trial_y_full,2), nPoiFail2),:)',...
    'Color', [0.5, 0.5, 0.5, 0.3])
medVal = median(bit_test(setdiff(1:size(trial_y_full,2), nPoiFail2),:));
plot(medVal,'r', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 5.5])
xticks(1:5)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dPoi-(5)','sCMP-(5,3)','sPoi-(5)'})
title('test')





