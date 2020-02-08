% RC circuit

rng(15)

Ts = 0.1;
no_obs = 1000;
R = 0.1^2;
Q = 0.02^2;

Cq = 0.8;
Rq = 1.6;

A = -1/Rq/Cq;
B = 1;
D = 0;
C = 1/Cq;

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1,1);
Bd = H(1,2);

q = NaN(1,no_obs+1);
y = NaN(1,no_obs);
u = NaN(1,no_obs+1);
q(1) = 0;
u(1) = 1;



for t=1:no_obs
%     u(t+1) = (2*(rand < 0.9)-1) * u(t);
    if rand > 0.9 
        u(t+1) = ~u(t);
    else
        u(t+1) = u(t);
    end

    
    q(t+1) = Ad*q(t) + Bd*u(t) + sqrt(Q)*randn;
    y(t) = C*q(t) + D*u(t)+sqrt(R)*randn;
    
end

q(:,end) = [];
u(end) = [];


%%
figure(1)
clf
plot(q)
hold on
plot(u)
plot(y,'o')
hold off

% Estimate the model using all the estimation data
noEstimationData = floor(0.67 * noObservations);
noValidationData = noObservations - noEstimationData;
y_estimation = y(1:noEstimationData);
y_validation = y(noEstimationData:end);
u_estimation = u(1:noEstimationData);
u_validation = u(noEstimationData:end);

save('../data/rc_circuit.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','Rq','Cq','c_true',...
    'R','n_states','Q','Ts')














