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

% This script generates the data to be used for system identifaction for
% example 2 and 4 in the paper. 

rng(44)                 % for reproduceability

clear all;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Specify Experiment Conditions
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N    = 400;         % Number of data samples
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

[bq,fq] = c2dm(num,den,Ts,'zoh');
Btrue = bq; Ftrue = fq;


u = zeros(1,N);  % The exogenous input 
u = [zeros(1,10),ones(1,round(N/6)), zeros(1,round(N/6)),zeros(1,round(N/6)),ones(1,round(N/6)),zeros(1,round(N/6))]; u = [u,zeros(1,N-length(u))];
R = 1e-4*eye(1);
z = filter(Btrue, Ftrue, u);
y = z + sqrt(R) * randn(size(z));




%% tf estimates
y_estimation = y(1:round(N/2));
u_estimation = u(1:round(N/2));

y_validation = y(round(N/2):end);
u_validation = u(round(N/2):end);

data_estimation = iddata(y_estimation.', u_estimation.');
m1 = oe(data_estimation, [7 6 0]);
f_ml = m1.f;
b_ml = m1.b;
sig_e1 = sqrt(m1.NoiseVariance);


data_estimation = iddata(y_estimation.', u_estimation.');
opt = oeOptions;
opt.Regularization.Lambda = 0.5;
m2 = oe(data_estimation, [11 10 0], opt);

f_ml2 = m2.f;
b_ml2 = m2.b;
sig_e2 = sqrt(m2.NoiseVariance);


tf_true = tf(Btrue,Ftrue,Ts);
%%
figure(3)
bode(tf_true)
hold on
bode(m1)
bode(m2)

%%
f_true = Ftrue;
b_true = Btrue;


save('../data/example2_and_4.mat','y_estimation','u_estimation',...
'Ts','u_validation','y_validation','f_ml2','b_ml2','sig_e2','f_true','b_true','f_ml','b_ml','sig_e1')


