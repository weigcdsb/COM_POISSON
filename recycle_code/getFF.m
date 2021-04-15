
function ff = getFF(y,type)

if nargin<2, type=''; end

switch lower(type)
    case 'np_bootstrap'
        [ff,bootsam] = bootstrp(500,@(y) var(y)/mean(y),y);
    case 'bayes_bootstrap'
        theta = rdirichlet(500,repmat(1,1,length(y)));
        wm = y'*theta;
        wv = sum(theta.*(bsxfun(@minus,y,wm).^2));
        ff = wv./wm;
    otherwise
        ff = var(y)/mean(y);
end