function [x,fx,dfdx,xx] = newtonGH(fdf,x0,TolX,MaxIter)

TolFun=eps;
xx(:,1) = x0;
dh = feval(fdf, x0);
fx = dh{1};
% disp(norm(fx))
warning('off');
for k = 1:MaxIter
%     disp(k)
    dfdx = dh{2};
    dx = -dfdx\fx;
    xx(:,k+1) = xx(:,k)+dx;
%     dhPre = dh;
    dh = feval(fdf,xx(:,k+1));
    fx = dh{1};
    if(norm(fx)<TolFun || norm(dx) < TolX)
        break;
    end
    
%     
%     if(norm(fx) > 1e4*dhPre{1})
%        x = nan;
%        dfdx = nan;
%        fprintf('skip');
%        return; 
%     elseif(norm(fx)<TolFun || norm(dx) < TolX)
%         break;
%     end
    
end
warning('on');

x = xx(:,k + 1);
dfdx = dh{2};
if(k == MaxIter)
    fprintf('The best in %d iterations\n',MaxIter)
end

end