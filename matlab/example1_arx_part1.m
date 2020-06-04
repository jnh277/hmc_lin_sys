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

% Auto-regressive Exogenous (ARX) model example part 1
% This script generates the data to be used for system identification and
% uses the matlab function 'arx' to determine the maximum likelihood
% estimate. For this part of the arx example the model order is assumed
% known.
%
% The data generated with this script is saved and then used by
% example1_arx_part1.py

rng(15)                 % for reproduceability
no_obs = 1000;          % number of observations, this will be split into estiamtion and validation later
Ts = 1;                 % time step
sig_e = 1;              % measurement noise standard deviation 
Band = [0 1];           % input band 

% define the system  

A = [1  -1.5  0.7];     % output coefficients   
B = [0 1 0.5];          % input coefficients
m0 = idpoly(A,B);       % creates a model of arx form

% generate random binary signal
u = iddata([], idinput(no_obs, 'rbs',Band));

% generate system noise
e = iddata([], sig_e*randn(no_obs, 1),'Ts',Ts);

% simulate system
y = sim(m0,[u e]);
z = [y, u];


dataIn = u.InputData;
dataOutNoisy = y.OutputData;

% split the data into estimation and validation sets
no_est = floor(0.67 * no_obs);      
no_val = no_obs - no_est;
y_estimation = dataOutNoisy(1:no_est);
y_validation = dataOutNoisy(no_est+1:end);
u_estimation = dataIn(1:no_est);
u_validation = dataIn(no_est+1:end);
estimationData = iddata(dataOutNoisy(1:no_est), dataIn(1:no_est));
validationData = iddata(dataOutNoisy(no_est+1:end), dataIn(no_est+1:end));

%% Estimate the model when the model order is known
modelEstimate = arx(estimationData, [length(m0.A)-1 length(m0.B) 0]);
K = 1;      % number of time steps ahead to predict
yhatOracle = predict(modelEstimate, validationData, K);        % by default this gives one step ahead predictions
y_hat_val_ML = yhatOracle.OutputData;

MF_ML = 100*(1 - sum((y_validation-y_hat_val_ML).^2)/sum(y_validation.^2));


%% save data
a_ML = modelEstimate.a;
b_ML = modelEstimate.b;
sig_e_ML = sqrt(modelEstimate.NoiseVariance);
a_true = m0.A;
b_true = m0.B;

save('../data/arx_example_part_one.mat','y_estimation', 'u_estimation', 'y_validation',...
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

hold off
legend('True sys','Maximum likelihood estimate')


