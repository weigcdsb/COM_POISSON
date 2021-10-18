function theta_ho = theta_interp1(theta_tr, trIdx, hoIdx)

theta_ho = zeros(size(theta_tr,1), length(hoIdx));
for d = 1:size(theta_tr,1)
    theta_ho(d,:) = interp1(trIdx,theta_tr(d,:),hoIdx);
end
if(isnan(theta_ho(:,end))); theta_ho(:,end) = theta_tr(:,end);end

end