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

% this script generates the data and runs the maximum likelihoodestiamtion
% for example 3 "fir model" given in Section 6.4 of the paper.
% the hmc system identification can then be run using example3_fir.py

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

% Estimate without regularisation but with an estimated model order of 13
m_arx = arx(eData, [0 13 0]);
yhat = predict(m_arx, vData);
yhat_arx = yhat.OutputData;

b_ML0 = m_arx.b;

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

[step_true,~] = step(trueSys);

MF_ML = 100*(1 - sum((y_validation(36:end)-y_hat_val_ML(36:end)).^2)/sum(y_validation(36:end).^2));
MF_ML0 = 100*(1 - sum((y_validation(36:end)-yhat_arx(36:end)).^2)/sum(y_validation(36:end).^2));
save('../data/example2_fir.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','y_hat_val_ML','b_ML','sig_e_ML',...
    'sig_e','nb','a_true','b_true','b_ML0','MF_ML','MF_ML0','g_true','step_true')

figure(1)
clf
plot(y_validation(36:end))
hold on
plot(yhat_arx(36:end),'.-')
plot(y_hat_val_ML(36:end),'.-')
hold off
legend('validation data','arx','arx TC')
%%
figure(2)
clf
plot(t,g_true)
hold on
plot(0:length(b_ML)-1,b_ML)
plot(0:length(b_ML0)-1,b_ML0)
hold off
legend('True','ARX TC','ARX')

figure(3)
clf
step(trueSys)
hold on
step(mrtc)
step(m_arx)
legend('true','ARX TC', 'ARX')