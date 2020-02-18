


ad = 0.9248;
bd = 0.0962;
c = 1.0;

P = 1.0;
Q = 0.02^2;

N = 100;

ts = 0:N-1;


K_check = P*(ad.^ts).'*(ad.^ts);


for i = 2:N
    K_check(i:end,i:end) = K_check(i:end,i:end)+ Q*ad.^(0:N-i).'*ad.^(0:N-i);
    
end
SQ = sqrt(Q);

Sk = zeros(N,N);
Sk(1,:) = ad.^ts*sqrt(P);
for i = 2:N
   Sk(i,i:end) = SQ*ad.^(0:N-i);  
    
end
K = Sk.'*Sk;
[Q,RV] = qr([Sk]);

% Sk = zeros(N,N);
% Sk(1,:) = ad.^ts;
% for i = 2:N
%    Sk(i,i:end) = ad.^(0:N-i);  
%     
% end
% [Q,R] = qr([diag(SV)*Sk]);

% 
L = RV(1:N,:).';        % this is a valid cholesky factor

% xs = real(sqrtm(K))*randn(N,1);
xs = L*randn(N,1);
upper = 2*sqrt(diag(K));
lower = -2*sqrt(diag(K));
% upper = 2*abs(diag(L));       % can't use the diagonal of cholesky factor
% for the variances... it isn't the same
% lower =-2*abs(diag(L));

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

% could we condition on direct measurements of a state
ts2 = [99,10];
R = 0.01^2;
x2 = [0.5;0.5];

Kg = K(ts2+1,:);
K22 = Kg(:,ts2+1) + eye(length(ts2))*R;
K21 = Kg;

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

%% ok so what if we have some  inputs
u = zeros(N,1);
u(1) = 1;
mu = zeros(N,1);
for t=1:N
%     u(t+1) = (2*(rand < 0.9)-1) * u(t);
    if rand > 0.9 
        u(t+1) = ~u(t);
    else
        u(t+1) = u(t);
    end
    if t < 2
        mu(t) = bd*u(t);
    else
        mu(t) = bd*u(t) + ad*mu(t-1);
    end
end


figure(3)
clf
xs = mu+ L*randn(N,1);
upper = mu+2*sqrt(diag(K));
lower = mu-2*sqrt(diag(K));
% upper = 2*abs(diag(L));       % can't use the diagonal of cholesky factor
% for the variances... it isn't the same
% lower =-2*abs(diag(L));

figure(1)
clf
plot(mu)
hold on
plot(upper,'r--')
plot(lower,'r--')
plot(xs)
hold off
legend('mean','upper CI','lower CI','sample')
title('random draw from prior with inputs')

% and conditioning on a measurement
ts2 = [80,10, 5, 20, 30];
R = 0.01^2;
% x2 = [1.0;0.2];
x2 = mu(ts2+1);

Kg = K(ts2+1,:);
K22 = Kg(:,ts2+1) + eye(length(ts2))*R;
K21 = Kg;

xhat = mu + K21.'*(K22\(x2-mu(ts2+1)));
Khat = K - K21.'*(K22\K21);
upper = xhat + real(sqrt(diag(Khat)));
lower = xhat - real(sqrt(diag(Khat)));

xs = xhat + real(sqrtm(Khat)) * randn(N,1); 

figure(4)
clf
plot(xhat)
hold on
plot(ts2+1,x2,'o')
plot(upper,'r--')
plot(lower,'r--')
plot(xs)
plot(mu,'-.')
hold off
title('Posterior with inputs')
legend('mean','upper CI','lower CI','sample')



