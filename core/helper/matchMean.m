
% Given a fixed nu, adjust lamda to achieve a desired mean
function lam = matchMean(mean_des,nu)

lam = nu*0;
for i=1:length(mean_des)
    err = @(p) (mean_des(i)-getMeanVar(p,nu(i))).^2;
    lam(i) = fminsearch(err,1,[]);
end
