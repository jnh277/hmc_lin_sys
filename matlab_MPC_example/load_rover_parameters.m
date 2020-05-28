clear all
clc

params = load('../results/rover_parameter_traces.mat');
%%
m = params.mass;
l = params.length;
J = params.inertia;
LQ = params.LQ;
LR = params.LR;
Q = nan(size(LQ));
R = nan(size(LR));

for i = 1:length(LQ)
   Q(i,:,:) = squeeze(LQ(i,:,:))*squeeze(LQ(i,:,:)).';
   R(i,:,:) = squeeze(LR(i,:,:))*squeeze(LR(i,:,:)).';
   
    
    
end

%% random subseelection of samples
n_sub = 500;
inds = randsample(length(m),n_sub);


%%
figure(1)
clf
subplot 331
histogram(m,'Normalization','pdf')
xlabel('mass')
ylabel('pdf')
hold on
histogram(m(inds),'Normalization','pdf')
hold off

subplot 332
plot(l,m,'.')
ylabel('mass')
xlabel('length')
hold on
plot(l(inds),m(inds),'.')
hold off

subplot 333
plot(J,m,'.')
ylabel('mass')
xlabel('inertia')
hold on
plot(J(inds),m(inds),'.')
hold off

subplot 334
plot(m,l,'.')
xlabel('mass')
ylabel('length')
hold on
plot(m(inds),l(inds),'.')
hold off


subplot 335
histogram(l,'Normalization','pdf')
xlabel('length')
ylabel('pdf')
hold on
histogram(l(inds),'Normalization','pdf')
hold off


subplot 336
plot(J,l,'.')
ylabel('length')
xlabel('inertia')
hold on
plot(J(inds),l(inds),'.')
hold off


subplot 338
plot(l,J,'.')
xlabel('length')
ylabel('inertia')
hold on
plot(l(inds),J(inds),'.')
hold off


subplot 337
plot(m,J,'.')
xlabel('mass')
ylabel('inertia')
hold on
plot(m(inds),J(inds),'.')
hold off

subplot 339
histogram(J,'Normalization','pdf')
xlabel('inertia')
ylabel('pdf')
hold on
histogram(J(inds),'Normalization','pdf')
hold off

% figure(2)
% clf
% histogram(sqrt(Q(:,1,1)),'Normalization','pdf')
