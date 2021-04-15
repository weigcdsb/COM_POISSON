
function theta = rdirichlet(n,alpha)

for i=1:n
    theta(:,i) = gamrnd(alpha,1);
end
theta = bsxfun(@rdivide,theta,sum(theta));