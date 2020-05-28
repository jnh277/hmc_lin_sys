function [f_sum,g_sum] = newCost_hmc(u,x0,xr,param)


ax = param.ax;
r1 = param.r1;
r2 = param.r2;
    


N = length(u);

sQ = diag([1 1 1 1e-4 1e-4]);
sR = diag([sqrt(0.1) sqrt(0.1)]);

x    = x0;
%dxdu = zeros(5,1);


f_sum = zeros(1,1);
g_sum = zeros(2*N,1);

M = length(param.m_hmc);
for i =1:M
    m  = param.m_hmc(i);
    l  = param.l_hmc(i);
    J  = param.J_hmc(i);
    Jpml2i = 1/(J + m*l*l);
    e  = zeros(N*7,1);
    if nargout > 1
        J  = zeros(N*7,2*N);
        dx = zeros(5,2*N);
    end
    for t=1:N
        x5m    = x(5)*Jpml2i;
        x      = x + 0.1*[cos(x(3))*x(4)/m;
                          sin(x(3))*x(4)/m;
                          x5m;
                          -r1*x(4)/m - m*l*(x5m)^2 + u(1,t)+u(2,t);
                          (l*x(4)-r2)*x5m+ax*u(1,t)-ax*u(2,t)];
        e((t-1)*7+1:t*7) = [sQ*(x-xr(:,t)); sR*u(:,t)];

        

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
    f_sum = f_sum + 0.5*(e.'*e)/M;
    g_sum = g_sum+ J'*e/M;
end
end