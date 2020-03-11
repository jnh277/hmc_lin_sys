clear all
clc

% how to create a random ARX model
default_test_model = true;
noObservations = 1000;
Ts = 1;

input_order_guess = 13;

rng(15)

% adjust noise standard deviation to be appropriate for the different
% models

sig_e = 0.1;    % works well with the default 
Band = [0 0.9];

% sig_e = 1;   % works well with rng(15) and n_states = 5
% Band = [0 0.3];

% sig_e = 0.01;   % works well with rng(15) and n_states = 3
% Band = [0 0.1];

% sig_e = 0.05;   % works well with rng(15) and n_states = 5
% Band = [0 0.3];


if default_test_model
    
    m0 = idtf([0.02008 0.04017 0.02008], [1 -1.561 0.6414], 1);
    n_states = 2;
    
else % generate a random test model
    n_states = 2;
    n_inputs = 1;
    n_outputs = 1;
    
    % generate random discrete time state space model with simple output
    % error noise structure
%     ss_sys = drss(n_states, n_outputs, n_inputs);
    m0 = rss(n_states, n_outputs, n_inputs);
    m0 = c2d(m0,Ts);
    m0 = idss(m0);

end

% generate random binary signal
u = iddata([], idinput(noObservations, 'rbs',Band));

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

% Estimate the model using ML
modelEstimate = arx(estimationData, [0 input_order_guess 0]);
yhatOracle = predict(modelEstimate, validationData);
y_hat_val_ML = yhatOracle.OutputData;

%% save data
b_ML = modelEstimate.b;
sig_e_ML = sqrt(modelEstimate.NoiseVariance);

a_true = m0.den;
b_true = m0.num;
save('../data/fir_order2.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','y_hat_val_ML','b_ML','sig_e_ML',...
    'sig_e','n_states','a_true','b_true')

%%
figure(1)
clf
plot(y_validation)
hold on
plot(y_hat_val_ML)

hold off
legend('Validation data','ML Predicted')


figure(3)
clf
bode(m0)
hold on
bode(modelEstimate)
bode([1, zeros(1,12)],b_ML,'Ts',1)
hold off


