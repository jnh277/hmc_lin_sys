% exploring the idea of horsehose
clear all
clc

num_dims = 1;
sigma = 1.0;

N = 10000;

X = sigma*randn(num_dims,N);
[F, XI] = ksdensity(X);
d = sqrt(sum(X.^2,1));

figure(2)
clf
subplot 211
histogram(X,'Normalization','pdf')
hold on
plot(XI, F,'LineWidth',2)
hold off
title('Samples drawn from a univariate Normal distribution')
xlabel('x')
ylabel('pdf')


% [F, XI] = ksdensity(d);
subplot 212
histogram(d,'Normalization','pdf')
% hold on
% plot(XI, F,'LineWidth',2)
% hold off
title('distance of samples from mean')
xlabel('distance')
ylabel('pdf')


%%
num_dims = 2;
sigma = 1.0;

% N = 5000;

X = sigma*randn(num_dims,N);

d = sqrt(sum(X.^2,1));
[c,edges] = histcounts(d);

% CI = 0.67;
CI = 0.95;

[v_sorted,i_sorted] = sort(c,'descend');
db = edges(i_sorted);
tt = cumsum(v_sorted) < CI * N;
min_d = min(db(tt));
max_d = max(db(tt));

inds_edge = logical((edges < max_d) .* (edges > min_d));





inds = logical((d > min_d) .* (d < max_d));

figure(3)
subplot 221
scatter(X(1,:),X(2,:))
axis equal
title('samples drawn from a bivariate Normal distribution')
xlabel('x_1')
ylabel('x_2')

subplot 222
histogram(X(1,:),'Normalization','pdf')
title('marginal pdf of x_1')
xlabel('x_1')
ylabel('pdf')

subplot 223
histogram(X(2,:),'Normalization','pdf')
title('marginal pdf of x_2')
xlabel('x_2')
ylabel('pdf')

subplot 224
% histogram(d,'Normalization','pdf')
histogram(d)
hold on
h = histogram(d(inds),edges(inds_edge));
hold off
title('distance of samples form mean')
xlabel('distance')
ylabel('number of samples')

figure(4)
clf
scatter(X(1,~inds),X(2,~inds))
hold on
scatter(X(1,inds),X(2,inds))
hold off

%% now draw a horseshoe prior
P = abs(trnd(1.0,1,N));
X = P.*randn(num_dims,N);
d = sqrt(sum(X.^2,1));

[c,edges] = histcounts(d,'BinLimits',[0 10]);

CI = 0.67;
% CI = 0.95;
[v_sorted,i_sorted] = sort(c,'descend');
db = edges(i_sorted);
tt = cumsum(v_sorted) < CI * N;
min_d = min(db(tt));
max_d = max(db(tt));

inds = logical((d > min_d) .* (d < max_d));


figure(5)
clf
subplot 221
scatter(X(1,:),X(2,:))
axis equal
xlim([-10 10])
ylim([-10 10])
title('Samples drawn from bivariate horse shoe prior')
xlabel('x1')
ylabel('x2')

subplot 222
histogram(X(1,:),'Normalization','pdf','BinLimits',[-2,2])
title('marginal pdf of x_1')
xlabel('x_1')
ylabel('pdf')

subplot 223
histogram(X(2,:),'Normalization','pdf','BinLimits',[-2,2])
title('marginal pdf of x_2')
xlabel('x_2')
ylabel('pdf')



subplot 224
histogram(d,'Normalization','pdf','BinLimits',[0,2])
title('distance of samples form mean')
xlabel('distance')
ylabel('pdf')

figure(6)
clf
scatter(X(1,~inds),X(2,~inds))
hold on
scatter(X(1,inds),X(2,inds))
hold off
xlim([-10 10])
ylim([-10 10])