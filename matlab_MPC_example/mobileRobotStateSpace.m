function [tm,xm] = mobileRobotStateSpace(tr,x0,u,param)

[tm,xm] = ode45(@(t,x) robotDynamics(t,x,u,param), tr , x0, odeset('RelTol',1e-6));

    function dx = robotDynamics(t,x,u,param)

    m  = param.m;
    l  = param.l;
    J  = param.J;
    ax = param.ax;
    r1 = param.r1;
    r2 = param.r2;
    
    Jpml2i = 1/(J + m*l*l);
    x5m    = x(5)*Jpml2i;
    dx = [cos(x(3))*x(4)/m;
          sin(x(3))*x(4)/m;
          x5m;
          -r1*x(4)/m - m*l*(x5m)^2 + u(1)+u(2);
          (l*x(4)-r2)*x5m+ax*u(1)-ax*u(2)]; 
    end
end
    






































%     
% %     G  = [0 0; 0 0; 0 0; 1 1; ax -ax];
% %     Q  = [cos(x(3)) 0; sin(x(3)) 0; 0 1];
% %     D  = [r1 0;0 r2];
% %     %M  = [m 0;0 J+m*l*l];
% %     C  = [0 -m*l*x(5)/(J+m*l*l); 
% %           m*l*x(5)/(J+m*l*l)  0];
% %     F  = [zeros(3) Q;-Q.' C-D];
% %     
% %     dx = F*[0;0;0;x(4)/m;x(5)/(J+m*l*l)] + G*u;
%     
%     
% end
% 
% end