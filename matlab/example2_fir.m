%##########################################################################
% Practical Bayesian System identification using Hamiltonian Monte Carlo
% Copyright (C) 2018  Johan Dahlin < uni (at) johandahlin [dot] com >
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%##########################################################################

% Load data from MATLAB help
load regularizationExampleData.mat eData

% Set random seed
rng(54531445)

% Taken from MATLAB documentation 
% "Regularized Identification of Dynamic Systems"
trueSys = idtf([0.02008 0.04017 0.02008], [1 -1.561 0.6414], 1);
[g_true, t] = impulse(trueSys);

% split into estimation and validation
data = eData;
eData = data(1:667);
vData = data(668:end);


% Estimate impulse response using regulatisation
nb = 35;

[L, R] = arxRegul(eData, [0 nb 0], arxRegulOptions('RegularizationKernel', 'TC'));
aopt = arxOptions;
aopt.Regularization.Lambda = L;
aopt.Regularization.R = R;
mrtc = arx(eData, [0 nb 0], aopt);
yhat_tc = predict(mrtc, vData);
yhat_tc = yhat_tc.OutputData;
[ghat_tc, ~, ~, ghat_tc_sd] = impulse(mrtc, t);

% Estimate impulse response using estimated model order
m = arx(eData, [0 13 0]);
yhat = predict(m, vData);
yhat = yhat.OutputData;



y_hat_val_ML = yhat_tc;
b_ML = mrtc.b;
sig_e_ML = sqrt(mrtc.NoiseVariance);

a_true = trueSys.den;
b_true = trueSys.num;

y_estimation = eData.OutputData;
u_estimation = eData.InputData;
y_validation = vData.OutputData;
u_validation = vData.InputData;
sig_e = 0.05;


save('../data/example2_fir.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','y_hat_val_ML','b_ML','sig_e_ML',...
    'sig_e','nb','a_true','b_true')

