function [f,g] = newCost_wlogBar(u,x0,xr,param)

m  = param.m;
l  = param.l;
J  = param.J;
ax = param.ax;
r1 = param.r1;
r2 = param.r2;
    
Jpml2i = 1/(J + m*l*l);

N = length(u);

sQ = diag([1 1 1 1e-4 1e-4]);
sR = diag([sqrt(0.1) sqrt(0.1)]);

x    = x0;
%dxdu = zeros(5,1);

e  = zeros(N*7,1);
if nargout > 1
    J  = zeros(N*7,2*N);
    dx = zeros(5,2*N);
end

fbsum = 0;
gbsum = 0;


for t=1:N
    x5m    = x(5)*Jpml2i;
    x      = x + 0.1*[cos(x(3))*x(4)/m;
                      sin(x(3))*x(4)/m;
                      x5m;
                      -r1*x(4)/m - m*l*(x5m)^2 + u(1,t)+u(2,t);
                      (l*x(4)-r2)*x5m+ax*u(1,t)-ax*u(2,t)];
    e((t-1)*7+1:t*7) = [sQ*(x-xr(:,t)); sR*u(:,t)];
    
    
    [fb,gb] = logBarrierRadial(2.5,5,5,10,x(1),x(2));
    fbsum = fbsum +fb;
    gbsum = gbsum +gb;
    
    if nargout > 1
        du               = zeros(2,2*N);
        du(1,2*(t-1)+1)  = 1;
        du(2,2*t)        = 1;
        A                = [0, 0, (-sin(x(3))*x(4))/m cos(x(3))/m, 0;
                            0, 0, cos(x(3))*x(4)/m sin(x(3))/m, 0;
                            0, 0, 0, 0, Jpml2i;
                            0, 0, 0, -r1/m,  -2*m*l*Jpml2i*x5m;
                            0, 0, 0, l*x5m, -r2*Jpml2i];
        B                = [0, 0;
                            0, 0;
                            0, 0;
                            1, 1;
                            ax, -ax];
        dx               = dx+0.1*[A*dx + B*du];

        J((t-1)*7+1:t*7,:) = [sQ*dx;sR*du];
        
        
    end
end


f = 0.5*(e.'*e) +100*fbsum;
if nargout >= 2
g = J'*e+gb;
end
end