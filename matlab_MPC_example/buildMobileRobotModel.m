function [param] = buildMobileRobotModel()

% Symbolic variables
syms q1 q2 q3
q = [q1 q2 q3].';
syms p1 p2
p = [p1 p2].';

% Constant variables (chosen arbitrarily)
param.m  = 20;
param.J  = 15;
param.l  = 0.5;
param.ax = 0.5;
param.r1 = 10;
param.r2 = 10;
param.C  = @(p) [0 -param.m*param.l*p(2)/(param.J+param.m*param.l^2); 
                 param.m*param.l*p(2)/(param.J+param.m*param.l^2) 0];
param.D  = @(q,p) [param.r1 0; 0 param.r2];
param.M  = @(q) [param.m 0; 0 param.J+param.m*param.l^2];
param.G  = @(q) [1 1; param.ax -param.ax];
param.Q  = @(q) [cos(q(3)) 0;
                 sin(q(3)) 0;
                 0         1];
param.V       = @(q) 0;
param.H       = @(q,p) 0.5*p.'*(param.M(q)\p) + param.V(q);
param.dHdx    = matlabFunction(jacobian(param.H(q,p),[q;p]),'vars',[q; p]);
param.dHdxCAT = @(q,p) param.dHdx(q(1),q(2),q(3),p(1),p(2));