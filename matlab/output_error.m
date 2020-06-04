clear all
clc
rng(15)


no_obs = 1000;
Ts = 0.025;
sig_e = 0.1;

sys = tf(0.6667,[1, 1.467, 2]);
sys_d = c2d(sys,Ts);
m0 = idpoly(sys_d);
Band = [0 0.3];
Range = [-10 10];

B = m0.B;
F = m0.F;




u = iddata([], idinput(no_obs, 'rbs',Band,Range),'Ts',Ts);

% generate system noise
e = iddata([], sig_e*randn(no_obs, 1),'Ts',Ts);

y = sim(m0,[u e]);
z = [y, u];

figure(1)
clf
plot(y)


% confirm understanding of output error models
yhat = zeros(no_obs,1);
w_est = nan(no_obs,1);
f = nan(no_obs,1);
e_save = zeros(no_obs,1);
% e_save(1:2) = e.inputData(1:2);        % require knowledge of the first two errors for closed loop
for i = 3:no_obs
    U = u.inputData(i:-1:i-2);
    Y = y.outputData(i-1:-1:i-2);
    

%     f(i) = B*U - F(2:3)*yhat(i-1:-1:i-2) - F(2:3)*e_save(i-1:-1:i-2);
%     yhat(i) = f(i);
%     e_save(i) = yhat(i) - y.outputData(i);
    
    %     This is a good estimate of
%     the mean however it is openloop
%     f(i) = B*U - F(2:3)*yhat(i-1:-1:i-2);   

%     e_save(i) = yhat(i) - y.outputData(i);
    Y = y.outputData(i:-1:i-2);
    e_save(i) =F*Y - B*U - F(2:3)*e_save(i-1:-1:i-2);
    
    
end

yhat2 = zeros(no_obs,1);
ehat = zeros(no_obs,1);
% ehat(1:2) = +e.inputData(1:2); 
% B2 = (0.4*rand(1,3)+0.8).*B;
% F2 = (0.4*rand(1,3)+0.8).*F;
F2 = F;
B2 = B;
% F2 = [1,fliplr([-0.45212922, -0.53738585])];
% B2 = fliplr([ 1.51851902e-03,  7.78928058e-05, -4.94978730e-04]);
for i = 3:no_obs
    U = u.inputData(i:-1:i-2);
    Y = y.outputData(i-1:-1:i-2);
    %     a self correcting estimate (too fucking good at correcting lol)
%     yhat2(i) = B2*U - F2(2:3)*yhat2(i-1:-1:i-2) + F2(2:3)*ehat(i-1:-1:i-2);  
%     ehat(i) = -yhat(i)+y.outputData(i);
    yhat2(i) = B2*U - F2(2:3)*Y + F2*e.inputData(i:-1:i-2);  

end


figure(1)
hold on
% plot([1:no_obs]*Ts,yhat,'r')
plot([1:no_obs]*Ts,yhat2,'g-.')
hold off


%%
% Estimate the model using all the estimation data
dataIn = u.InputData;
dataOutNoisy = y.OutputData;

noEstimationData = floor(0.67 * no_obs);
noValidationData = no_obs - noEstimationData;
y_estimation = dataOutNoisy(1:noEstimationData);
y_validation = dataOutNoisy(noEstimationData:end);
u_estimation = dataIn(1:noEstimationData);
u_validation = dataIn(noEstimationData:end);
f_coef_true = F;
b_coef_true = B;

data_estimation = iddata(y_estimation, u_estimation);
data_validation = iddata(y_validation, u_validation);

% Estimate model using known model orders
m1 = oe(data_estimation, [3 2 0]);
yhat = predict(m1, data_validation);
yhat = yhat.OutputData;

F_ML = m1.F;
B_ML = m1.B;

% save('../data/oe_order2.mat','y_estimation', 'u_estimation', 'y_validation',...
%     'u_validation','f_coef_true','b_coef_true',...
%     'sig_e','Ts','F_ML','B_ML')


%% can i do maximum likelihood like this
% phi = [];
% for i = 3:no_obs
%     phi = [phi;-y.outputData(i-1:-1:i-2).',u.inputData(i:-1:i-2).'];
%     
%     
% end
% 
% mu = y.outputData(3:end);
% phi\mu
%     

%% 
Bt = fliplr([-0.00014906,  0.00100474, -0.000441  ]);
Ft = [1, fliplr([ 0.96299131, -1.9617489 ])];

t = tf(Bt,Ft,Ts);

figure(5)
bode(t)
hold on
bode(sys_d)
hold off
    
    


