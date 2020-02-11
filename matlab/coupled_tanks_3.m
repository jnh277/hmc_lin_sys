% sim a series RLC circuit
clear all
clc

% RC circuit

rng(15)

Ts = 0.2;
no_obs = 1000;
R = 0.1^2;
Q = 0.02^2;

C1 = 0.8;
C2 = 3.3;
C3 = 2.2;

R12 = 0.5;
R13 = 1.2;
R23 = 2.3;

Ro = 3.8;



A = [-(1/R12+1/R13)/C1, 1/R12/C2, 1/R13/C3;
    1/R12/C1, -(1/R12+1/R23)/C2, 1/R23/C3;
    1/R13/C1, 1/R23/C2, -(1/R13+1/R23+1/Ro)/C3];

B = [1;0;0];
D = 0;
C = [1/C1, 0, 0;
    0, 0, 1/C3];

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1:3,1:3);
Bd = H(1:3,4);

q = NaN(3,no_obs+1);
y = NaN(2,no_obs);
u = NaN(1,no_obs+1);
q(:,1) = [0,0,0];
u(1) = 1;



for t=1:no_obs
    if rand > 0.95 
        u(t+1) = ~u(t);
    else
        u(t+1) = u(t);
    end

    
    q(:,t+1) = Ad*q(:,t) + Bd*u(t) + sqrt(Q)*randn(3,1);
    y(:,t) = C*q(:,t) + D*u(t)+sqrt(R)*randn(2,1);
    
end

q(:,end) = [];
u(end) = [];


%%
figure(1)
clf
plot(q(3,:))
hold on
plot(u)
plot(y(2,:),'o')
hold off


figure(2)
clf
plot(q(1,:))
hold on
plot(q(2,:))
plot(q(3,:))
hold off

% Estimate the model using all the estimation data
noObservations = no_obs;
noEstimationData = floor(0.67 * noObservations);
noValidationData = noObservations - noEstimationData;
y_estimation = y(:,1:noEstimationData);
y_validation = y(:,noEstimationData:end);
u_estimation = u(1:noEstimationData);
u_validation = u(noEstimationData:end);
states_est = q(:,1:noEstimationData);
states_val = q(:,noEstimationData:end);

save('../data/c_tanks3.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','C1','C2','C3','R12','R13','R23','Ro',...
    'R','Q','Ts','states_est','states_val')














