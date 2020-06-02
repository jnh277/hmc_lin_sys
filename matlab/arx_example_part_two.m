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

% Auto-regressive Exogenous (ARX) model example part 2
% This script generates the data to be used for system identification and
% uses the matlab function 'arx' to determine the maximum likelihood
% estimate. For this part of the arx example the model order is unknown and
% is chosen by picking the model with the highest model fit factor on the
% validation data set
%
% The data generated with this script is saved and then used by
% arx_example_part_two.py


clear all
clc

% arx data with outlier measurements using student T
rng(15) % for repeatability
no_obs = 1000;
Ts = 1;

sig_e = 1;    % works well with the default 
Band = [0 1];


    
A = [1  -1.5  0.7];
B = [0 1 0.5];
m0 = idpoly(A,B);
n_states = 1;
    

% generate random binary signal

u = iddata([], idinput(no_obs, 'rbs',Band));

%% generate system noise
meas_errors = sig_e*randn(no_obs, 1);
e = iddata([], meas_errors,'Ts',Ts);

%% simulate system
y = sim(m0,[u e]);
z = [y, u];


dataIn = u.InputData;
dataOutNoisy = y.OutputData;

% split data intro training and validation
noEstimationData = floor(0.67 * no_obs);
noValidationData = no_obs - noEstimationData;
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
MF = zeros([10 11]);
for na=1:10
    for nb=1:11
        modelEstimate = arx(estimationData, [na nb 0]);
%         predictionErrObject = pe(modelEstimate, validationData);
%         predictionError(na, nb) = sum((predictionErrObject.OutputData).^2);
        yhatOracle = predict(modelEstimate, validationData, 1);        % by default this gives one step ahead predictions
        y_hat_val_ML = yhatOracle.OutputData;
        MF(na,nb) = 100*(1 - sum((y_validation-y_hat_val_ML).^2)/sum(y_validation.^2));
%         disp([na nb predictionError(na, nb)]);
    end
    disp(na)
end

% Find the model order that minimises the squared prediction error
% idx = find(min(min(predictionError)) == predictionError);
% [na_found, nb_found] = ind2sub([10 11], idx);

idx = find(max(max(MF)) == MF);
[na_found, nb_found] = ind2sub([10 11], idx);


% Estimate the model using found order and  all the estimation data

modelEstimate = arx(estimationData, [na_found nb_found 0]);
yhatOracle = predict(modelEstimate, validationData);
y_hat_val_ML = yhatOracle.OutputData;



%% save data
a_ML = modelEstimate.a;
b_ML = modelEstimate.b;
sig_e_ML = sqrt(modelEstimate.NoiseVariance);
a_true = m0.A;
b_true = m0.B;


MF_ML = MF(na_found,nb_found);

save('../data/arx_example_part_two.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','y_hat_val_ML','a_ML','b_ML','sig_e_ML','a_true','b_true',...
    'sig_e','MF_ML')

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
% set(gca,'LineWidth',2)


