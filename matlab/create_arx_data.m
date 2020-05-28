clear all
clc

% a hard system t = tf(rand(1,5),[1 41/10 92/5 134/5 105/2 5])

% how to create a random ARX model
default_test_model = true;
noObservations = 1000;
Ts = 1;

rng(15)

% adjust noise standard deviation to be appropriate for the different
% models

% sig_e = 2;    % works well with the default 
% Band = [0 1];

% sig_e = 1;   % works well with rng(15) and n_states = 5
% Band = [0 0.3];

% sig_e = 0.01;   % works well with rng(15) and n_states = 3
% Band = [0 0.1];

sig_e = 0.05;   % works well with rng(15) and n_states = 5
Band = [0 0.3];


if default_test_model
    
    A = [1  -1.5  0.7];
    B = [0 1 0.5];
    m0 = idpoly(A,B);
    n_states = 2;
    
else % generate a random test model
    n_states = 4;
    n_inputs = 1;
    n_outputs = 1;
    
    % generate random discrete time state space model with simple output
    % error noise structure
%     ss_sys = drss(n_states, n_outputs, n_inputs);
    ss_sys = rss(n_states, n_outputs, n_inputs);
    A_true = ss_sys.A;
    B_true = ss_sys.B;
    C_true = ss_sys.C;
    D_true = ss_sys.D;
    figure(2)
    bode(ss_sys)
    
    ss_sys = c2d(ss_sys,Ts);
    
    % convert to idss structure taht allows non simple error model
    ss_sys = idss(ss_sys);
    
    
    % add a 'K' term for non simple noise on states, otherwise when using
    % idpoly will get an output error model rather than an arx model
    ss_sys.K = -0.5+1*rand(n_states,1);
    
    % now convert to arx
    m0 = idpoly(ss_sys);

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

% Estimate the model when the model order is known
modelEstimate = arx(estimationData, [length(m0.A)-1 length(m0.B) 0]);
yhatOracle = predict(modelEstimate, validationData);
y_hat_val_ML = yhatOracle.OutputData;

%% save data
a_ML = modelEstimate.a;
b_ML = modelEstimate.b;
sig_e_ML = sqrt(modelEstimate.NoiseVariance);
a_true = m0.A;
b_true = m0.B;

% save('../data/arx_order4.mat','y_estimation', 'u_estimation', 'y_validation',...
%     'u_validation','y_hat_val_ML','a_ML','b_ML','sig_e_ML','a_true','b_true',...
%     'sig_e','n_states','A_true','B_true','C_true','D_true')

%%
figure(1)
clf
plot(y_validation)
hold on
plot(y_hat_val_ML)

hold off
legend('Validation data','ML Predicted')


