

ad = 0.95;
c = 1.0;

P = 1.0;
mu = 0;     % for now this should be kept the case
Q = 0.02^2;

N = 100;

ts = 0:N-1;


K = P*(ad.^ts).'*(ad.^ts);
for i = 2:N
    K(i:end,i:end) = K(i:end,i:end)+ Q*ad.^(0:N-i).'*ad.^(0:N-i);
    
end
% K = [P, P*ad, P*Ad^2;
%     ad*P, ad*P*ad,   
%     ad*ad*P, ad*ad*P*ad, ad*ad*P*ad*ad



xs = real(sqrtm(K))*randn(N,1);
upper = 2*sqrt(diag(K));
lower =-2*sqrt(diag(K));

figure(1)
clf
plot(zeros(N,1))
hold on
plot(upper,'r--')
plot(lower,'r--')
plot(xs)
hold off
legend('mean','upper CI','lower CI','sample')
title('random draw from prior')

% could we condition on say the first state being 1?
ts2 = [99,10];
R = 0.01^2;
x2 = [0.5;0.5];

Kg = K(ts2+1,:);
K22 = Kg(:,ts2+1) + eye(length(ts2))*R;
K21 = Kg;
% K21(:,ts2+1) = [];

% for i = length(ts2)
%    for j = length(ts2)
%        K22(i,j) = K(ts2(i)+1,ts2(j)+1);
%    end
% end

% Q SHOULD BE IN THE ABOVE SOMEWHERE?

xhat = K21.'*(K22\x2);
Khat = K - K21.'*(K22\K21);
upper = xhat + real(sqrt(diag(Khat)));
lower = xhat - real(sqrt(diag(Khat)));

xs = xhat + real(sqrtm(Khat)) * randn(N,1); 

figure(2)
clf
plot(xhat)
hold on
plot(ts2+1,x2,'o')
plot(upper,'r--')
plot(lower,'r--')
plot(xs)
hold off
title('Posterior')
legend('mean','upper CI','lower CI','sample')




