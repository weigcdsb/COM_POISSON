addpath(genpath('C:\Users\gaw19004\Documents\GitHub\COM_POISSON'));
% addpath(genpath('D:\github\COM_POISSON'));

usr_dir = 'C:\Users\gaw19004';
r_path = 'C:\Users\gaw19004\Documents\R\R-4.0.2\bin';
r_wd = [usr_dir '\Documents\GitHub\COM_POISSON\core\runRcode'];

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

nModel = 7;
LLHD = ones(size(trial_y_full,2), nModel + 1)*NaN;
LLHD_ho = ones(size(trial_y_full,2), nModel + 1)*NaN;

LLHD_spk = [];
BIT_spk = [];
% SSIDX = []; 
load('ssidx.mat')% use selected from previous...

for n = failIdx' % n = 1:size(trial_y_full,2)
    
    LLHD_spk{n} = NaN*ones(2,nModel + 1);
    BIT_spk{n} = NaN*ones(2,nModel);
    c = 0;
    flag = false;
    while(sum(isnan(LLHD_spk{n}),'all') > 0 && c < 10)
        c = c + 1;
        if flag
            disp('sampling')
            ssIdx = zeros(size(data.EVENTS, 2)*nSS, 1);
            for k = 1:size(data.EVENTS, 2)
                ssIdx(((k-1)*nSS + 1):(k*nSS)) =...
                    sort(randsample(((k-1)*nAll+1): k*nAll,nSS));
            end
            SSIDX{n} = ssIdx;
        end
        
        try
            disp("iter:" + n)
            [~,LLHD_spk{n}, BIT_spk{n}, LLHD(n,:), LLHD_ho(n,:)] =...
                evalc('models_run_na(n, trial_x_full, trial_y_full,SSIDX{n}, nSS, nAll, usr_dir, r_path, r_wd)');
            flag = false;
        catch
            flag = true;
            disp('error: resample...')
        end
        
    end
    
end


%% let's plot
% failIdx = find(isnan(sum(LLHD, 2)));
% failIdx = find(LLHD(:,4) < LLHD(:,8));
% failIdx = find(LLHD(:,4) < LLHD(:,7) | LLHD(:,1) < LLHD(:,4));
% failIdx = [failIdx; find(isnan(sum(LLHD, 2)))];

bit_mat = cell2mat(BIT_spk');
bit_train = bit_mat(1:2:end,:);
bit_test = bit_mat(2:2:end,:);

bit_train2 = zeros(size(bit_train));
bit_test2 = zeros(size(bit_test));
% bit/trial
for k = 1:size(bit_train, 1)
    spkTmp = sum(trial_y_full(SSIDX{k}, k));
    spkTmp_ho = sum(trial_y_full(setdiff(1:length(trial_x_full), SSIDX{k}), k));
    
    bit_train2(k,:) = spkTmp*bit_train(k,:)/size(data.EVENTS,2);
    bit_test2(k,:) = spkTmp_ho*bit_test(k,:)/size(data.EVENTS,2);
end

bit_train = bit_train(setdiff(1:size(trial_y_full,2), failIdx),:);
bit_test = bit_test(setdiff(1:size(trial_y_full,2), failIdx),:);



figure(1)
subplot(2,1,1)
hold on
plot(bit_train','Color', [0.5, 0.5, 0.5, 0.3])
q1 = quantile(bit_train,0.25);
medVal = median(bit_train);
q3 = quantile(bit_train,0.75);
plot(medVal,'r', 'LineWidth',2)
plot(q1, 'r--', 'LineWidth',2)
plot(q3, 'r--', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.25,medVal(t)+0.35,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
%   text(t-0.25,medVal(t)+0.015,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 7.5])
xticks(1:7)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
    'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})
title('bit/spk-train')

subplot(2,1,2)
hold on
plot(bit_test','Color', [0.5, 0.5, 0.5, 0.3])

q1 = quantile(bit_test,0.25);
medVal = median(bit_test);
q3 = quantile(bit_test,0.75);
plot(medVal,'r', 'LineWidth',2)
plot(q1, 'r--', 'LineWidth',2)
plot(q3, 'r--', 'LineWidth',2)
for t = 1:numel(medVal)
  text(t-0.25,medVal(t)+0.35,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
end
hold off
xlim([0.5 7.5])
xticks(1:7)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
    'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})
title('bit/spk-test')

%% boxplot
subplot(2,1,1)
boxplot(bit_train)
xticks(1:7)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
    'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})
subplot(2,1,2)
boxplot(bit_test)
xticks(1:7)
xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
    'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})







% bit_train2 = zeros(size(bit_train));
% bit_test2 = zeros(size(bit_test));
% % bit/trial
% for k = 1:size(bit_train, 1)
%     spkTmp = sum(trial_y_full(SSIDX{k}, k));
%     spkTmp_ho = sum(trial_y_full(setdiff(1:length(trial_x_full), SSIDX{k}), k));
%     
%     bit_train2(k,:) = spkTmp*bit_train(k,:)/size(data.EVENTS,2);
%     bit_test2(k,:) = spkTmp_ho*bit_test(k,:)/size(data.EVENTS,2);
% end



% figure(2)
% subplot(2,1,1)
% hold on
% plot(bit_train2','Color', [0.5, 0.5, 0.5, 0.3])
% q1 = quantile(bit_train2,0.25);
% medVal = median(bit_train2);
% q3 = quantile(bit_train2,0.75);
% plot(medVal,'r', 'LineWidth',2)
% plot(q1, 'r--', 'LineWidth',2)
% plot(q3, 'r--', 'LineWidth',2)
% for t = 1:numel(medVal)
%   text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
% end
% hold off
% xlim([0.5 7.5])
% xticks(1:7)
% xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
%     'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})
% title('bit/trial-train')
% 
% subplot(2,1,2)
% hold on
% plot(bit_test2','Color', [0.5, 0.5, 0.5, 0.3])
% 
% q1 = quantile(bit_test2,0.25);
% medVal = median(bit_test2);
% q3 = quantile(bit_test2,0.75);
% plot(medVal,'r', 'LineWidth',2)
% plot(q1, 'r--', 'LineWidth',2)
% plot(q3, 'r--', 'LineWidth',2)
% for t = 1:numel(medVal)
%   text(t-0.3,medVal(t)+0.3,num2str(round(medVal(t),3)),'Color','red','FontSize',15)
% end
% hold off
% xlim([0.5 7.5])
% xticks(1:7)
% xticklabels({'dCMP-(5,3)','dCMP-(5,1)','dCMP-(5)-nu','dPoi-(5)',...
%     'sCMP-(5,3)','sCMP-(5,1)','sPoi-(5)'})
% title('bit/trial-test')


%% compares...

% dCMP(5,3) vs. sCMP(5,3)
% 1 & 5
% figure(3)
% subplot(1,2,1)
% d1_train = [bit_train(:, [1 5]) bit_train(:,1) - bit_train(:,5)];
% d1_train_x = repelem(1:3, size(d1_train, 1));
% % boxplot(d1_train)
% beeswarm(d1_train_x',d1_train(:),'sort_style','square',...
%     'dot_size',.5,'overlay_style','box');
% xticks(1:3)
% xticklabels({'dCMP-(5,3)','sCMP-(5,3)','diff: d - s'})
% title('bit/spk-train')
% 
% subplot(1,2,2)
% d1_test = [bit_test(:, [1 5]) bit_test(:,1) - bit_test(:,5)];
% d1_test_x = repelem(1:3, size(d1_test, 1));
% beeswarm(d1_test_x',d1_test(:),'sort_style','square',...
%     'dot_size',.5,'overlay_style','box');
% xticks(1:3)
% xticklabels({'dCMP-(5,3)','sCMP-(5,3)','diff: d - s'})
% title('bit/spk-test')
% sgtitle('dCMP-(5,3) vs. sCMP-(5,3)')

% figure(4)
% subplot(1,3,1)
% d1_train = bit_train(:, [1 5]);
% d1_train_x = repelem(1:2, size(d1_train, 1));
% % boxplot(d1_train)
% beeswarm(d1_train_x',d1_train(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'dCMP-(5,3)','sCMP-(5,3)'})
% title('train')
% 
% subplot(1,3,2)
% d1_test = bit_test(:, [1 5]);
% d1_test_x = repelem(1:2, size(d1_test, 1));
% beeswarm(d1_test_x',d1_test(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'dCMP-(5,3)','sCMP-(5,3)'})
% title('test')
% 
% subplot(1,3,3)
% d1_diff = [bit_train(:,1) - bit_train(:,5) bit_test(:,1) - bit_test(:,5)];
% d1_diff_x = repelem(1:2, size(d1_diff, 1));
% beeswarm(d1_diff_x',d1_diff(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'train','test'})
% title('diff: dCMP-(5,3) - sCMP-(5,3)')
% sgtitle('dCMP-(5,3) vs. sCMP-(5,3): bit/spk')
% set(gcf,'Position',[0 0 800 500])



% dCMP(5,1) vs. sCMP(5,const)
% 2 & 3
% figure(5)
% subplot(1,2,1)
% d2_train = [bit_train(:, [2 3]) bit_train(:,2) - bit_train(:,3)];
% d2_train_x = repelem(1:3, size(d2_train, 1));
% % boxplot(d1_train)
% beeswarm(d2_train_x',d2_train(:),'sort_style','square',...
%     'dot_size',.5,'overlay_style','box');
% xticks(1:3)
% xticklabels({'dCMP-(5,1)','dCMP-(5,const)','diff: d_nu - const'})
% title('bit/spk-train')
% 
% subplot(1,2,2)
% d2_test = [bit_test(:, [2 3]) bit_test(:,2) - bit_test(:,3)];
% d2_test_x = repelem(1:3, size(d2_test, 1));
% beeswarm(d2_test_x',d2_test(:),'sort_style','square',...
%     'dot_size',.5,'overlay_style','box');
% xticks(1:3)
% xticklabels({'dCMP-(5,1)','dCMP-(5,cont)','diff: d_nu - const'})
% title('bit/spk-test')
% sgtitle('dCMP-(5,1) vs. dCMP-(5,const)')

% figure(6)
% subplot(1,3,1)
% d2_train = bit_train(:, [2 3]);
% d2_train_x = repelem(1:2, size(d2_train, 1));
% beeswarm(d2_train_x',d2_train(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:3)
% xticklabels({'dCMP-(5,1)','dCMP-(5,const)'})
% title('train')
% 
% subplot(1,3,2)
% d2_test = bit_test(:, [2 3]);
% d2_test_x = repelem(1:2, size(d2_test, 1));
% beeswarm(d2_test_x',d2_test(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'dCMP-(5,1)','dCMP-(5,const)'})
% title('test')
% 
% subplot(1,3,3)
% d2_diff = [bit_train(:,2) - bit_train(:,3) bit_test(:,2) - bit_test(:,3)];
% d2_diff_x = repelem(1:2, size(d2_diff, 1));
% beeswarm(d2_diff_x',d2_diff(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'train','test'})
% title('diff: dCMP-(5,1) - dCMP-(5,const)')
% sgtitle('dCMP-(5,1) vs. dCMP-(5,const): bit/spk')
% set(gcf,'Position',[0 0 800 500])
% 
% figure(7)
% subplot(1,3,1)
% d3_train = bit_train(:, [2 6]);
% d3_train_x = repelem(1:2, size(d3_train, 1));
% % boxplot(d1_train)
% beeswarm(d3_train_x',d3_train(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'dCMP-(5,1)','sCMP-(5,1)'})
% title('train')
% 
% subplot(1,3,2)
% d3_test = bit_test(:, [2 6]);
% d3_test_x = repelem(1:2, size(d3_test, 1));
% beeswarm(d3_test_x',d3_test(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'dCMP-(5,1)','sCMP-(5,1)'})
% title('test')
% 
% subplot(1,3,3)
% d3_diff = [bit_train(:,2) - bit_train(:,6) bit_test(:,2) - bit_test(:,6)];
% d3_diff_x = repelem(1:2, size(d3_diff, 1));
% beeswarm(d3_diff_x',d3_diff(:),'sort_style','square',...
%     'dot_size',1,'overlay_style','box');
% xticks(1:2)
% xticklabels({'train','test'})
% title('diff: dCMP-(5,1) - sCMP-(5,1)')
% sgtitle('dCMP-(5,1) vs. sCMP-(5,1): bit/spk')
% set(gcf,'Position',[0 0 800 500])





