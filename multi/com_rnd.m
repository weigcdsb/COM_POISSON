function x = com_rnd(lambda, nu, N)

if nargin<3
    for i=1:length(lambda)
        cdf = cumsum(com_pdf(0:1000,lambda(i),nu(i)));
        y = rand(1);
        [tmp,x(i)] = histc(y,cdf);
    end
elseif length(N)==1
    cdf = cumsum(com_pdf(0:1000,lambda,nu));
    x = rand(N,1);
    [tmp,x] = histc(x,cdf);
end