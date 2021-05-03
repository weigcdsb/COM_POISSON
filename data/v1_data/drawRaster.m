function drawRaster(firings,isSub, doCol, tRange)

if nargin<2, isSub=0; end
if nargin<3, doCol=0; end
if nargin<4, tRange=[-Inf Inf]; end

if any(doCol)
    if length(doCol)==1
        if doCol>0
            if iscell(firings)
                cmap=hsv(length(firings));
            else
                cmap=lines(max(firings(:,2)));
            end
        else
            cmap=0;
        end
    elseif length(doCol)==3
        cmap(1:length(firings),:) = repmat(doCol,length(firings),1);
        doCol=1;
    else
        uCol = unique(doCol);
%         cmap0 = lines(length(uCol));
        cmap0 = cbrew(length(uCol)+1,'paired_div');
        cmap  = zeros(length(doCol),3);
        for i=1:length(uCol)
            cmap(doCol==uCol(i),:) = repmat(cmap0(i,:),sum(doCol==uCol(i)),1);
        end
        doCol=1;
    end
end

% tic
% for i=1:size(firings,1)
%     line([firings(i,2) firings(i,2)],[firings(i,1) firings(i,1)+1]);
% end
% toc

if ~isSub
    clf
end

if iscell(firings)
    C = length(firings);
    hold on
    %     for i=1:size(firings,2)
    %        plot(firings{i},i,'.','MarkerSize',1)
    %     end
    for i=1:C
        firings{i} = firings{i}(firings{i}<tRange(2));
        firings{i} = firings{i}(firings{i}>tRange(1));
        if size(firings{i},2)>size(firings{i},1), firings{i}=firings{i}'; end
        if ~isempty(firings{i})
            if doCol>0
                line([firings{i} firings{i}]',repmat([i-1 i],size(firings{i},1),1)','Color',cmap(i,:))
            elseif doCol<0
                plot(firings{i}',i,'.','Color','k')
            else
                line([firings{i} firings{i}]',repmat([i-1 i],size(firings{i},1),1)','Color','k')
            end
        end
    end
    hold off
else
    firings=firings(firings(:,1)<tRange(2),:);
    firings=firings(firings(:,1)>tRange(1),:);
    C = max(firings(:,2));
    %     plot(firings(:,1),firings(:,2),'.','MarkerSize',1)
    for i=1:size(firings,1)
        if doCol
            line([firings(i,1) firings(i,1)],[firings(i,2)-1 firings(i,2)],'Color',cmap(firings(i,2),:))
        elseif doCol<0
            plot(firings(i,1),firings(i,2),'.','Color',cmap(i,:))
        else
            line([firings(i,1) firings(i,1)],[firings(i,2)-1 firings(i,2)],'Color','k')
        end
    end
end

% set(gca,'YTick',[])
% step = 10;
% set(gca,'YTick',[0.5-step:step:C-0.5])
% ylab=[''];
% for i=0:step:C
%     ylab = [ylab '|' num2str(i)];
% end
% set(gca,'YTickLabel',ylab)

axis tight
xlim(tRange)
ylim([0 C])