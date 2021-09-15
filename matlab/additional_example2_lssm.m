%##############################################################################
%    Practical Bayesian System Identification using Hamiltonian Monte Carlo
%    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <https://www.gnu.org/licenses/>.
%##############################################################################

% An additional example not included in the paper
% this script generates the data and runs the maximum likelihoodestiamtion
% for a lienar state space system
% the hmc system identification can then be run using additonal_example2_lssm.py


clear all
clc

rng(15)


Ts = 0.01;
no_obs = 3000;
R = 0.01^2;
Q = diag([0, 0, 0, 0.1^2]);

n_states = 4;
n_inputs = 1;
n_outputs = 1;

% generate random discrete time state space model with simple output
% error noise structure
ss_sys = rss(n_states, n_outputs, n_inputs);
A = ss_sys.A;
B = ss_sys.B;
C = ss_sys.C;
D = ss_sys.D;

figure(1)
bode(ss_sys)

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1:n_states,1:n_states);
Bd = H(1:n_states,n_states+1);

% discretising Q
F = expm([-A, Q;
    zeros(size(A)), A.']*Ts);

Qd = F(n_states+1:2*n_states,n_states+1:2*n_states).'*F(1:n_states,n_states+1:2*n_states);       % transpose of bottom right multipled by top right
LQ = chol(Qd,'lower');

fs = 1/Ts;
fmax = fs/2;        % (Hz) this is the fastest frequency that can be recovered
w_max = fmax*2*pi;

q = NaN(n_states,no_obs+1);
y = NaN(1,no_obs);
u = zeros(1,no_obs+1);
q(:,1) = zeros(n_states,1);



no_sins = 20;
T = Ts*(1:no_obs);
omegas = [0,logspace(-1,log10(w_max/2),no_sins-1)];
u = sum(sin(omegas.'*T + rand(no_sins,1)),1);


for t=1:no_obs    
    q(:,t+1) = Ad*q(:,t) + Bd*u(t) + sqrtm(Qd)*randn(n_states,1);
    y(t) = C*q(:,t) + D*u(t)+sqrt(R)*randn;    
end

% q(:,end) = [];
% u(end) = [];


%%
% figure(2)
% clf
% plot(q(1,:))
% hold on
% plot(u)
% plot(y,'o')
% hold off
% 
% 
% figure(3)
% clf
% plot(q(1,:))
% hold on
% plot(q(2,:))
% hold off

sys = ss(A,B,C,D);


% Estimate the model using all the estimation data
noObservations = no_obs;
noEstimationData = floor(1.0 * noObservations);
noValidationData = noObservations - noEstimationData;
y_estimation = y(1:noEstimationData);
y_validation = y(noEstimationData+1:end);
u_estimation = u(1:noEstimationData);
u_validation = u(noEstimationData+1:end);
states_est = q(:,1:noEstimationData);
states_val = q(:,noEstimationData+1:end);

A_true = A;
B_true = B;
C_true = C;
D_true = D;

dat = iddata(y_estimation.',u_estimation.',Ts);
[sys_ML,x0_ML] = ssest(dat,4);

A_ML = sys_ML.A;
B_ML = sys_ML.B;
C_ML = sys_ML.C;
D_ML = sys_ML.D;

%% simulate maximum likelihood system
H = expm([A_ML, B_ML;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad_ML = H(1:n_states,1:n_states);
Bd_ML = H(1:n_states,n_states+1);
y_ML2 = y;

q_ML = q;
for t=1:no_obs    
    q_ML(:,t+1) = Ad_ML*q_ML(:,t) + Bd_ML*u(t);
    y_ML2(t) = C_ML*q_ML(:,t) + D_ML*u(t);    
end

x_ML = q_ML(:,1:end-1);

%%

figure(3)
clf
bode(sys_ML)
hold on
bode(sys)
hold off

save('../data/additional_example2_lssm.mat','y_estimation', 'u_estimation',...
    'A_true','B_true','C_true','D_true',...
    'R','Q','Ts','states_est','w_max','A_ML','B_ML','C_ML','D_ML','x_ML')


