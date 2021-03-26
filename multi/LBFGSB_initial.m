function [x, xhist] = LBFGSB_initial(y, x_lam, g_nu, dt)



func = @(x)com_llhd2(y, x(1), x(2), true);
% x0 = [mean(y); mean(y)/var(y)];
opts.x0 = [mean(y); mean(y)/var(y)];

[x,f,info] = lbfgsb(func, [0;0], [Inf;Inf], opts);




end
