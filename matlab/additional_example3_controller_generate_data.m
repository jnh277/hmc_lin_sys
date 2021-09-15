% ###############################################################################
% #    Practical Bayesian Linear System Identification using Hamiltonian Monte Carlo
% #    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
% #
% #    This program is free software: you can redistribute it and/or modify
% #    it under the terms of the GNU General Public License as published by
% #    the Free Software Foundation, either version 3 of the License, or
% #    any later version.
% #
% #    This program is distributed in the hope that it will be useful,
% #    but WITHOUT ANY WARRANTY; without even the implied warranty of
% #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% #    GNU General Public License for more details.
% #
% #    You should have received a copy of the GNU General Public License
% #    along with this program.  If not, see <https://www.gnu.org/licenses/>.
% ###############################################################################

% An additional example not included in the paper
% generates data to be used for the system identification of a 6th order
% linear state space system. This system identification is run using
% additional_example3_sysid.py

clear all;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Specify Experiment Conditions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N    = 200;         % Number of data samples
T    = 1;           % Sampling Period

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Specify a linear system
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% den11     = [1 1.1 0.1];
% den12     = [1 2.5 1];
% den21     = [1 1 0.21];
% den22     = [1 1.2 0.32];
% sysc      = tf({1,3; 1 1}, {den11, den12; den21, den22});
% sysd      = c2d(sysc,T,'zoh');
% [A,B,C,D] = ssdata(sysd); 

% sys=drss(3,1,1);
% [A,B,C,D]=ssdata(sys);
Ts = 1;
n=6; p=1; m=1;
den = real( poly([-0.1,-0.2,-0.02+j*1,-0.02-j*1,-0.01-j*0.1,-0.01+j*0.1]) );
num = 10*den(length(den));
[a,b,c,d] =tf2ss(num,den); 
sysc=ss(a,b,c,d); 
sys=c2d(sysc,Ts,'zoh');
[A,B,C,D]=ssdata(sys);

% sys=ss(c2d(tf([0 0 1],[1/(2*pi*5)^2 2*.05/(2*pi*5) 1]),0.01,'zoh'));
% systr = sys;
% [A,B,C,D]=ssdata(sys);

nx        = size(A,1);
ny        = size(C,1);
nu        = size(B,2);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Simulate a data record
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
R = 1e-4*eye(ny);
S = zeros(nx,ny);
Q = 1e-3*eye(nx);
u = randn(size(B,2),N);  % The exogenous input
e = (sqrtm(R)*randn(size(C,1),N)); % The measurement noise sequence
w = sqrtm(Q)*randn(nx,N);
x = zeros(nx,N+1); 
y = zeros(ny,N);
for t=1:N
	y(:,t)   = C*x(:,t) + D*u(:,t) + e(:,t);  %Simulate output
	x(:,t+1) = A*x(:,t) + B*u(:,t) + w(:,t);  %Simulate state with innovations structure
end
z.y = y; 
z.u = u;


%% estimate using ssest function

dat = iddata(y.',u.',Ts);
[sys_ML,x0_ML] = ssest(dat,6);

A_ML = sys_ML.A;
B_ML = sys_ML.B;
C_ML = sys_ML.C;
D_ML = sys_ML.D;
%% simulate maximum likelihood system
[n_states,~] = size(A_ML);
H = expm([A_ML, B_ML;zeros(size(B)).', zeros(size(B,6))]*Ts);
Ad_ML = H(1:n_states,1:n_states);
Bd_ML = H(1:n_states,n_states+1);
y_ML = y;

x_ML = x;
for t=1:N    
    x_ML(:,t+1) = Ad_ML*x_ML(:,t) + Bd_ML*u(t);
    y_ML(t) = C_ML*x_ML(:,t) + D_ML*u(t);    
end

x_ML = x_ML(:,1:end-1);

figure(3)
clf
plot(y)
hold on
plot(y_ML)
hold off

%%
figure(2)
clf
bode(sysc)
hold on
bode(sys_ML)
hold off


%%
y_estimation = y;
u_estimation = u;

save('../data/control_example_data.mat','a','b','c','d','x','y_estimation','u_estimation','x_ML','A_ML',...
    'B_ML','C_ML','D_ML','Ts')

% m.ss.A=A;
% m.ss.B=B;
% m.ss.C=C;
% m.ss.D=D;
% m.ss.Q=Q;
% m.ss.S=S;
% m.ss.R=R;
% m.ss.X1=zeros(nx,1);
% m.ss.P1=eye(nx);

%Run EM code to obtain initial estimate
% clear m
% m.nx=nx; m.type='ss'; o.alg='em';
% m=est(z,m,o);
% 
% bode(sys,m.sysG)
% drawnow;
% 
% %Set priors
% m.ss.Lam = eye(nx+ny);%[m.ss.Q m.ss.S;m.ss.S' m.ss.R]
% m.ss.ell = 1;
% m.ss.M   = [m.ss.A m.ss.B;m.ss.C m.ss.D];
% m.ss.V   = eye(nx+nu);
