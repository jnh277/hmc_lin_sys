% RC circuit

Ts = 0.1;
no_obs = 300;
R = 0.01^2;
Q = 0.001^2;

Cq = 0.8;
Rq = 1.3;

A = -1/Rq/Cq;
B = 1;
D = 0;
C = 1/Cq;

H = expm([A, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
Ad = H(1,1);
Bd = H(1,2);

q = NaN(2,no_obs+1);
y = NaN(1,no_obs);
u = zeros(1,no_obs);
q(:,1) = [0,0];
% u(1) = 1;

u(1:25) = 1;
u(100:200) = 1;





for t=1:no_obs
    
%     if rand > 0.9 
%         u(t+1) = ~u(t);
%     else
%         u(t+1) = u(t);
%     end
    
    
    q(1,t+1) = Ad*q(1,t) + Bd*u(t) + sqrt(Q)*randn;
    
    A2 = -1/Rq/(Cq*exp(10*q(1,t)));
    H2 = expm([A2, B;zeros(size(B)).', zeros(size(B,2))]*Ts);
    Ad2 = H2(1,1);
    Bd2 = H2(1,2);
    q(2,t+1) = Ad2*q(2,t) + Ad*q(1,t) + sqrt(Q)*randn;
    y(t) = C*q(2,t) + D*u(t)+sqrt(R)*randn;
    
end

q(:,end) = [];


%%
figure(1)
clf
plot(q(2,:))
hold on
plot(u)
plot(y,'o')
hold off

figure(2)
plot(q(1,:))

















