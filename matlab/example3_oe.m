% example 3 oe
% rng(54531445)
rng(13)
sig_e = 0.5;

% Generate data
noObservations = 1000;
u = randn(noObservations, 1);
[B, A] = cheby1(3, 5, 0.8);
y = filter(B, A, u);
y = y + sig_e * randn(length(y), 1);

% Partition into estimate and validation sets
no_est = floor(0.5*noObservations);
y_estimation = y(1:no_est);
u_estimation = u(1:no_est);
y_validation = y(no_est:end);
u_validation = u(no_est:end);
data_estimation = iddata(y_estimation, u_estimation);
data_validation = iddata(y_validation, u_validation);

% Estimate model using known model orders
m1 = oe(data_estimation, [4 3 0]);
yhat = predict(m1, data_validation);
yhat = yhat.OutputData;

% Estimate model using regularisation (without knowing model orders)
opt = oeOptions;
opt.Regularization.Lambda = 0.5;
m2 = oe(data_estimation, [11 10 0], opt);
yhat_reg = predict(m2, data_validation);
yhat_reg = yhat_reg.OutputData;
%%
MF_ML1 = 100*(1 - sum((y_validation(4:end)-yhat(4:end)).^2)/sum(y_validation(4:end).^2));
MF_ML2 = 100*(1 - sum((y_validation(11:end)-yhat_reg(11:end)).^2)/sum(y_validation(11:end).^2));

%%
f_ml = m1.f;
b_ml = m1.b;
sig_e_ML = sqrt(m1.NoiseVariance);
f_ml2 = m2.f;
b_ml2 = m2.b;
sig_e_ML2 = sqrt(m1.NoiseVariance);

f_true = A;
b_true = B;

save('../data/example3_oe.mat','y_estimation', 'u_estimation', 'y_validation',...
    'u_validation','yhat','f_ml','b_ml','f_ml2','b_ml2','sig_e_ML','sig_e_ML2',...
    'sig_e','f_true','b_true')
%% plot
figure(1)
clf
plot(y_validation)
hold on
plot(yhat)
plot(yhat_reg)
hold off
legend('True','known order','regularised')

trueSys = tf(B,A,1);


[MAG,PHASE,W] = bode(trueSys);
[MAG1,PHASE1,W1] = bode(m1);
[MAG2,PHASE2,W2] = bode(m2);
figure(2)
clf
subplot(2,1,1)
plot(W,squeeze(MAG(1,1,:)))
hold on
plot(W1,squeeze(MAG1))
plot(W2,squeeze(MAG2))
hold off
xlabel('Frequency')
ylabel('MAG')
legend('True','known order','regularised')

subplot(2,1,2)
plot(W,squeeze(PHASE))
hold on
plot(W1,squeeze(PHASE1)-360)
plot(W2,squeeze(PHASE2)-360)
hold off
xlabel('Frequency')
ylabel('Phase')
legend('True','known order','regularised')

