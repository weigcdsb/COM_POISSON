function theta_ho = theta_interp1(theta_tr, trIdx, hoIdx)

theta_ho = zeros(size(theta_tr,1), length(hoIdx));
for d = 1:size(theta_tr,1)
    theta_ho(d,:) = interp1(trIdx,theta_tr(d,:),hoIdx);
end

nanIdx = find(isnan(sum(theta_ho, 1)));
for k = 1:length(nanIdx)
    if(nanIdx(k) > length(hoIdx)/2)
        theta_ho(:,nanIdx(k)) = theta_tr(:,end);
    else
        theta_ho(:,nanIdx(k)) = theta_tr(:,1);
    end
end

end


%%
% 
% theta_tr = theta_fit1;
% trIdx = ssIdx;
% hoIdx = hoIdx;



