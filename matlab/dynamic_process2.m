

ad = 0.9;
c = 1.0;

P = 1.0;
mu = 0;     % for now this should be kept the case
Q = 0.0001;

N = 50;

ts = 0:N-1;


% K = P*(ad.^ts).'*(ad.^ts);
% for i = 2:N
%     K(i:end,i:end) = K(i:end,i:end)+ Q*ad.^(0:N-i).'*ad.^(0:N-i);
%     
% end
% K = [P, P*ad, P*ad^2;
%     ad*P, ad*P*ad+Q, P*ad+Q*ad;
%     ad*ad*P, ad*P+Q*ad, ad^2*P*ad^2+ad^2*Q+Q];

K = NaN(N,N);
for i = 1:N
   K(i,i:N) = P*ad.^(0:N-i);
   K(i:N,i) = P*ad.^(0:N-i).'; 
   K(i,i) = ad^(i-1)*P*ad^(i-1);
    
end

% for i = 2:N
%     K(i:end,i:end) = K(i:end,i:end)+ Q*ad.^(0:N-i).'*ad.^(0:N-i);
%     
% end




% K = [P, P*ad, P*Ad^2;
%     ad*P, ad*P*ad, 
%     ad*ad*P, ad*ad*P*ad, ad*ad*P*ad*ad

% K = K

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
% ts2 = [0,30];
% R = 0.01;
% K21 = ad.^ts2.'*P*ad.^ts;
% K21(2,31) = 1.0*Q;
% K22 = ad.^ts2.'*P*ad.^ts2+R*eye(length(ts2));
% % x2 = [1.0;0.5];
% 
% xhat = K21.'*(K22\x2);
% Khat = K - K21.'*(K22\K21);
% upper = xhat + real(sqrt(diag(Khat)));
% lower = xhat - real(sqrt(diag(Khat)));
% 
% 
% figure(2)
% clf
% plot(xhat)
% hold on
% plot(ts2+1,x2,'o')
% plot(upper,'r--')
% plot(lower,'r--')
% hold off




