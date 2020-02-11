% sim a series RLC circuit
clear all
clc

% RC circuit

rng(15)

Ts = 0.1;
no_obs = 1000;
R = 0.1^2;
Q = 0.02^2;

Cq = 0.8;
Rq = 1.6;
Lq = 2.2;

A = [-Rq/Lq, -1/Cq;1/Lq 0];
B = [1;0];
D = 0;
C = [0, 1/Cq];

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1:2,1:2);
Bd = H(1:2,3);

q = NaN(2,no_obs+1);
y = NaN(1,no_obs);
u = NaN(1,no_obs+1);
q(:,1) = [0,0];
u(1) = 1;



for t=1:no_obs
    if rand > 0.95 
        u(t+1) = ~u(t);
    else
        u(t+1) = u(t);
    end

    
    q(:,t+1) = Ad*q(:,t) + Bd*u(t) + sqrt(Q)*randn(2,1);
    y(t) = C*q(:,t) + D*u(t)+sqrt(R)*randn;
    
end

q(:,end) = [];
u(end) = [];


%%
figure(1)
clf
plot(q(2,:))
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

% Estimate the model using all the estimation data
noObservations = no_obs;
noEstimationData = floor(0.67 * noObservations);
noValidationData = noObservations - noEstimationData;
y_estimation = y(1:noEstimationData);
y_validation = y(noEstimationData:end);
u_estimation = u(1:noEstimationData);
u_validation = u(noEstimationData:end);
states_est = q(:,1:noEstimationData);

save('../data/rlc_circuit.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','Rq','Cq','Lq',...
    'R','Q','Ts','states_est')














