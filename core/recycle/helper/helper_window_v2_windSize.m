function neg_llhd_filt = helper_window_v2_windSize(windSize, Q, theta0, N, X_lam,...
    G_nu, W0, F, windType,searchStep)

[~,~, ~, ~, ~,...
    lam_filt,nu_filt,log_Zvec_filt] =...
    ppafilt_compoisson_v2_window_fisher(theta0, N, X_lam, G_nu, W0, F, Q, windSize*searchStep, windType);

% [~,~, lam_filt,nu_filt,log_Zvec_filt] =...
%     ppafilt_compoisson_v2_window_fisher(theta0, N, X_lam, G_nu, W0, F, Q, windSize*searchStep, windType);


if(length(log_Zvec_filt) == size(N, 2))
    llhd_filt = sum(N.*log((lam_filt+(lam_filt==0))) -...
        nu_filt.*gammaln(N + 1) - log_Zvec_filt);
    fprintf('llhd %.02f... \n', llhd_filt);
    neg_llhd_filt = -llhd_filt;
else
    neg_llhd_filt = Inf;
end


end