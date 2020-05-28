function [c,ceq] = newLinConstraintsHacky_hmc(u,x0,param)


ax = param.ax;
r1 = param.r1;
r2 = param.r2;
    

N = length(u);


M = length(param.m_hmc);
c = nan(N,M);
ceq = [];


for i =1:M
    m  = param.m_hmc(i);
    l  = param.l_hmc(i);
    J  = param.J_hmc(i);
    Jpml2i = 1/(J + m*l*l);

    x    = x0;
    for t=1:N
        x5m    = x(5)*Jpml2i;
        x      = x + 0.1*[cos(x(3))*x(4)/m;
                          sin(x(3))*x(4)/m;
                          x5m;
                          -r1*x(4)/m - m*l*(x5m)^2 + u(1,t)+u(2,t);
                          (l*x(4)-r2)*x5m+ax*u(1,t)-ax*u(2,t)];

        rad  = (x(1)-5).^2 + (x(2)-5.5).^2;
        c(t,i) = 2.5^2-rad;



    end
    
end
    c = max(c,2);
end