% generate data for an order 2 system

clear all
clc
noObservations = 1000;
sig_e = 0.7;
Ts = 0.1;

rng(15)

n_states = 1;
sys = c2d(tf([0 1],[1, 1, 5]),Ts);
m0 = idss(sys);
m0.K = [0.1;0.01];     % add in some state noise just for fun


% generate random binary signal
Band = [0 0.15];
Range = [-20,20];
u = iddata([], idinput(noObservations, 'rbs',Band,Range),'Ts',Ts);

% generate system noise
e = iddata([], sig_e*randn(noObservations, 1),'Ts',Ts);

% simulate system
y = sim(m0,[u e]);
z = [y, u];


dataIn = u.InputData;
dataOutNoisy = y.OutputData;

% Estimate the model using all the estimation data
noEstimationData = floor(0.67 * noObservations);
noValidationData = noObservations - noEstimationData;
y_estimation = dataOutNoisy(1:noEstimationData);
y_validation = dataOutNoisy(noEstimationData:end);
u_estimation = dataIn(1:noEstimationData);
u_validation = dataIn(noEstimationData:end);
estimationData = iddata(dataOutNoisy(1:noEstimationData), dataIn(1:noEstimationData));
validationData = iddata(dataOutNoisy(noEstimationData:end), dataIn(noEstimationData:end));

% % Estimate the model using ML
% modelEstimate = arx(estimationData, [0 input_order_guess 0]);
% yhatOracle = predict(modelEstimate, validationData);
% y_hat_val_ML = yhatOracle.OutputData;
% 
% %% save data
% b_ML = modelEstimate.b;
% sig_e_ML = sqrt(modelEstimate.NoiseVariance);

sys = idss(m0);
a_true = sys.a;
b_true = sys.b;
c_true = sys.c;
d_true = sys.d;


% save('../data/ss_order2.mat','y_estimation', 'u_estimation', 'y_validation',...
%     'u_validation','a_true','b_true','c_true','d_true',...
%     'sig_e','n_states')

%%
figure(1)
clf
plot(y_validation)
hold on
% plot(y_hat_val_ML)

hold off
legend('Validation data')
