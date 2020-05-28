function [c,ceq] = newLinConstraintsHacky(u,x0,param)

m  = param.m;
l  = param.l;
J  = param.J;
ax = param.ax;
r1 = param.r1;
r2 = param.r2;
    
Jpml2i = 1/(J + m*l*l);

N = length(u);


x    = x0;
%dxdu = zeros(5,1);

% e  = zeros(N*7,1);
% if nargout > 1
%     J  = zeros(N*7,2*N);
%     dx = zeros(5,2*N);
% end
c = nan(N,1);
ceq = [];


for t=1:N
    x5m    = x(5)*Jpml2i;
    x      = x + 0.1*[cos(x(3))*x(4)/m;
                      sin(x(3))*x(4)/m;
                      x5m;
                      -r1*x(4)/m - m*l*(x5m)^2 + u(1,t)+u(2,t);
                      (l*x(4)-r2)*x5m+ax*u(1,t)-ax*u(2,t)];
                  
    r1  = (x(1)-5).^2 + (x(2)-5.5).^2;
%     r2  = (x(1)-11).^2 + (x(2)-5).^2;
    c(t,1) = 2.5^2-r1;
%     c(t,2) = 2.5^2-r2;
%     c(t) = (x(1) - 5) .* (x(2) < 5);
    

end