%% Simulate a two state mass spring damper system

clear all
clc

% RC circuit

rng(15)

Ts = 0.025;
no_obs = 1000;
R = 0.1^2;
% Q = diag([0.02^2; 0.05^2]);
Q = diag([0, 0.1^2]);


Mq = 1.5;
Kq = 3.0;
Dq = 2.2;

A = [0, 1;-Kq/Mq, -Dq/Mq];
B = [0;1/Mq];
D = 0;
C = [1, 0];

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1:2,1:2);
Bd = H(1:2,3);

% discretising Q
F = expm([-A, Q;
    zeros(size(A)), A.']*Ts);

Qd = F(3:4,3:4).'*F(1:2,3:4);       % transpose of bottom right multipled by top right
LQ = chol(Qd,'lower');


% Nyquist sampling limit
fs = 1/Ts;
fmax = fs/2;        % (Hz) this is the fastest frequency that can be recovered
w_max = fmax*2*pi;

q = NaN(2,no_obs+1);
y = NaN(1,no_obs);
u = zeros(1,no_obs+1);
q(:,1) = [0,0];



% % staggered sins input
% T = Ts*(1:no_obs);
% u(1:340) = 2.0;
% u(341:560) = sin(0.141*T(341:560));
% u(561:780) = sin(1.41*T(561:780));
% u(781:1000) = sin(20*T(781:1000));

% % sum of sins input
no_sins = 6;
T = Ts*(1:no_obs);
omegas = [0,logspace(-1,log10(w_max/5),no_sins-1)];
u = sum(sin(omegas.'*T + rand(no_sins,1)),1);

%% observability test
%%

for t=1:no_obs
%     if rand > 0.975
%         u(t+1) = -u(t);
%     else
%         u(t+1) = u(t);
%     end

    
    q(:,t+1) = Ad*q(:,t) + Bd*u(t) + sqrtm(Qd)*randn(2,1);
    y(t) = C*q(:,t) + D*u(t)+sqrt(R)*randn;
    
end

% q(:,end) = [];
% u(end) = [];


%%
figure(1)
clf
plot(q(1,:))
hold on
plot(u)
plot(y,'o')
hold off


figure(2)
clf
plot(q(1,:))
hold on
plot(q(2,:))
hold off

sys = ss(A,B,C,D);
figure(3)
clf
bode(sys)

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

% save('../data/msd_Qfull.mat','y_estimation', 'u_estimation', 'y_validation',...
%     'u_validation','Mq','Kq','Dq','A_true','B_true','C_true','D_true',...
%     'R','Q','Ts','states_est','states_val')

save('../data/msd_sumsins.mat','y_estimation', 'u_estimation',...
    'Mq','Kq','Dq','A_true','B_true','C_true','D_true',...
    'R','Q','Ts','states_est','w_max')












