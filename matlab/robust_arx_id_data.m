clear all
clc

% arx data with outlier measurements using student T
rng(15) % for repeatability
noObservations = 1000;
Ts = 1;

sig_e = 2;    % works well with the default 
Band = [0 1];


    
A = [1  -1.5  0.7];
B = [0 1 0.5];
m0 = idpoly(A,B);
n_states = 2;
    

% generate random binary signal

u = iddata([], idinput(noObservations, 'rbs',Band));

%% generate system noise
meas_errors = sig_e*randn(noObservations, 1);
meas_errors_trnd = sig_e*trnd(1.2,noObservations,1);
figure(5)
clf
histogram(meas_errors,30,'Normalization','pdf')
hold on
histogram(meas_errors_trnd,30,'Normalization','pdf')

e = iddata([], meas_errors_trnd,'Ts',Ts);
% e = iddata([], meas_errors,'Ts',Ts);

%% simulate system
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

%% finding based on minimum prediction error
% Estimate the model when the model order is unknown
% Select model order by exhaustive search using half of the estimation set
% for estimating model and the remaining for computing the prediction error
predictionError = zeros([10 11]);
for na=1:10
    for nb=1:10
        modelEstimate = arx(estimationData, [na nb 0]);
        predictionErrObject = pe(modelEstimate, validationData);
        predictionError(na, nb) = sum((predictionErrObject.OutputData).^2);
%         disp([na nb predictionError(na, nb)]);
    end
    disp(na)
end

% Find the model order that minimises the squared prediction error
idx = find(min(min(predictionError)) == predictionError);
[na_found, nb_found] = ind2sub([10 10], idx);


% Estimate the model when the model order is known
modelEstimate = arx(estimationData, [length(m0.A)-1 length(m0.B) 0]);
yhatOracle = predict(modelEstimate, validationData);
y_hat_val_ML = yhatOracle.OutputData;

% Estimate the model using the found model order
modelEstimate_min = arx(estimationData, [na_found nb_found 0]);
yhatOracle_min = predict(modelEstimate, validationData);
y_hat_val_ML_min = yhatOracle.OutputData;

% Estimate using a too large model order
modelEstimate2 = arx(estimationData, [10 10 0]);
yhatOracle2 = predict(modelEstimate, validationData);
y_hat_val_ML2 = yhatOracle.OutputData;

% estimate using regularisation
Option = arxRegulOptions('RegularizationKernel', 'TC');
[Lambda, R] = arxRegul(estimationData, [10 10 0], Option);
arxOpt = arxOptions;
arxOpt.Regularization.Lambda = Lambda;
arxOpt.Regularization.R = R;
modelEstimateReg = arx(estimationData, [10 10 0], arxOpt);



%% save data
a_ML = modelEstimate.a;
b_ML = modelEstimate.b;
sig_e_ML = sqrt(modelEstimate.NoiseVariance);
a_ML_reg = modelEstimateReg.a;
b_ML_reg = modelEstimateReg.b;
sig_e_ML_reg = sqrt(modelEstimateReg.NoiseVariance);
a_ML_min = modelEstimate_min.a;
b_ML_min = modelEstimate_min.b;
sig_e_ML_min = sqrt(modelEstimate_min.NoiseVariance);
a_true = m0.A;
b_true = m0.B;



% save('../data/robust_noise_id_data1_3.mat','y_estimation', 'u_estimation', 'y_validation',...
%     'u_validation','a_ML','b_ML','sig_e_ML','a_true','b_true',...
%     'sig_e','a_ML_reg','b_ML_reg','sig_e_ML_reg','a_ML_min','b_ML_min','sig_e_ML_min')

%%
figure(1)
clf
plot(y_validation)
hold on
plot(y_hat_val_ML)

hold off
legend('Validation data','ML Predicted')

figure(2)
clf
bode(m0)
hold on
bode(modelEstimate)
% bode(modelEstimate2)
bode(modelEstimateReg)
bode(modelEstimate_min)
set(gca,'LineWidth',2)


