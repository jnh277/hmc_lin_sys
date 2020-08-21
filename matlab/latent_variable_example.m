% ###############################################################################
% #    Practical Bayesian Linear System Identification using Hamiltonian Monte Carlo
% #    Copyright (C) 2020  Johannes Hendriks < johannes.hendriks@newcastle.edu.a >
% #
% #    This program is free software: you can redistribute it and/or modify
% #    it under the terms of the GNU General Public License as published by
% #    the Free Software Foundation, either version 3 of the License, or
% #    any later version.
% #
% #    This program is distributed in the hope that it will be useful,
% #    but WITHOUT ANY WARRANTY; without even the implied warranty of
% #    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% #    GNU General Public License for more details.
% #
% #    You should have received a copy of the GNU General Public License
% #    along with this program.  If not, see <https://www.gnu.org/licenses/>.
% ###############################################################################

% This script is the associated matlab code to go with the illustrative
% example of jointly sampling the latent variables and parameters provided
% in Section 5.1 of the paper.

clear all
clc
% rng(6)
rng(10)
%% simulate system
a = 0.9;
x0 = 1.5;
sig_r = 0.05;
sig_q = 0.1;


N = 10;

x = NaN(1,N+1);
y = NaN(1,N);
x(1) = x0;

for k = 1:N
   x(k+1) = a* x(k) + sig_q*randn;
   y(k) = x(k) + sig_r * randn;
end




%% set up HMC
M = 0.2*eye(N+1);

% saved_cov = load('sample_cov');
% M = saved_cov.sample_cov;

sM = sqrtm(M);
detM = det(M);



process_mod = @(x_next,x, a) sum(-log(2*pi*sig_q^2) - 0.5*(x_next-a*x).^2/sig_q^2);
d_process_mod = @(x_next,x, a) [-(x_next - a*x)/sig_q^2;
                                a*(x_next-a*x)/sig_q^2;
                                x.'*(x_next-a*x)/sig_q^2];
dq_process_mod = @(q) [q(2:end-1).'*(q(3:end)-q(1)*q(2:end-1))/sig_q^2;
                        q(1)*(q(3:end)-a*q(2:end-1))/sig_q^2;
                        0] + [0;0;-(q(3:end) - q(1)*q(2:end-1))/sig_q^2];

                            
                            
meas_mod = @(x) sum(-log(2*pi*sig_r^2) -0.5*(y.'-x).^2/sig_r^2);
d_meas_mod = @(x) (y.'-x)/sig_r^2;

V = @(q) - process_mod(q(3:end),q(2:end-1),q(1)) - meas_mod(q(2:end,1));
dVdq = @(q) [0;-d_meas_mod(q(2:end))] - dq_process_mod(q);
            
H   = @(q,p) V(q) + 0.5*p.'*(M\p) + log(2*pi*detM);

num = 10000;     % number of iterations
q = NaN(N+1,num);       %[a, x_1, x_2, x_3, ... , x_N]
p   = zeros(N+1,num);
alp  = zeros(1,num);
err  = zeros(1,num);
% q(:,1) = [0.9;y.'];
q(:,1) = [0.8;y.'];
q(:,1) = [0.8;y.'.*(0.5+rand(N,1))];



acc = 0;

wb = waitbar(0,'HMC sampling');
for i=1:num
    %Step 1. propose p
    p(:,i) = sM*randn(N+1,1);
    %Integrate forwards and backwards in time for some time using leapfrog
    T   = 0.1;
    ee   = 0.001;
    nint  = floor(T/ee);
    qqf  = NaN(N+1,nint+1);
    ppf  = NaN(N+1,nint+1);
    qqf(:,1) = q(:,i);
    ppf(:,1) = p(:,i);
    

    for k=1:nint
        tmpf   = ppf(:,k) - (ee/2)*dVdq(qqf(:,k));
        qqf(:,k+1) = qqf(:,k) + ee*(M\tmpf);
        ppf(:,k+1) = tmpf  - (ee/2)*dVdq(qqf(:,k+1));
        
    end
    
    err(i) = H(q(:,i),p(:,i)) - H(qqf(:,k+1),-ppf(:,k+1));
    %Accept or reject and throw away momentum
    alp(i) = min(1,exp(err(i)));
    if rand<alp(i)
        q(:,i+1) = qqf(:,k+1);
        acc = acc + 1;
    else
        q(:,i+1) = q(:,i);
    end
    waitbar(i/num,wb)
end
delete(wb)
%% throw away warmup
q = q(:,floor(num)/2:end);


%%

q_95_upper = prctile(q,97.5,2);
q_95_lower = prctile(q,2.5,2);

q_65_upper = prctile(q,50+32.5,2);
q_65_lower = prctile(q,50-32.5,2);

q_mean = mean(q,2);

fontsize = 30;

figure(1)
clf

hold on

h95 = patch([1:N, N:-1:1],[q_95_lower(2:end).', fliplr(q_95_upper(2:end).')],[0, 0.4470, 0.7410],'FaceAlpha',0.2,'LineStyle','None');
h65 = patch([1:N, N:-1:1],[q_65_lower(2:end).', fliplr(q_65_upper(2:end).')],[0, 0.4470, 0.7410],'FaceAlpha',0.4,'LineStyle','None');

he = plot(q_mean(2:end),'Color',[0, 0.4470, 0.7410],'LineWidth',2);
ht = plot(x(1:N),'k','LineWidth',2);
hm = plot(y,'r*','LineWidth',2,'MarkerSize',10);

box on
set(gca,'FontSize',16,'LineWidth',1)
hold off
xlabel('Time step $t$','Interpreter','Latex','FontSize',fontsize)
ylabel('$p(x_k | y_{1:T})$','Interpreter','Latex','FontSize',fontsize)
hl = legend([ht,hm,he, h95, h65],'True state $x_k$','Measurenents $y_k$','Posterior mean','Posterior $95\%$ CI','Posterior $65\%$ CI');
set(hl,'Interpreter','Latex','FontSize',20)
xlim([1 N])
%%
[f,xi] = ksdensity(q(1,:));

figure(4)
clf
histogram(q(1,:),'Normalization','pdf')
ylimits = get(gca,'YLim');
hold on
h1 = plot([a, a],ylimits,'k--','LineWidth',3);
h2 = plot([q_mean(1), q_mean(1)],ylimits,'--','LineWidth',3);
% plot(xi,f,'Color',[0, 0.4470, 0.7410],'LineWidth',3)
hold off
set(gca,'FontSize',16,'LineWidth',1)
xlabel('$\theta$','Interpreter','Latex','FontSize',fontsize)
ylabel('$p(\theta|y_{1:T})$','Interpreter','Latex','FontSize',fontsize)
hl = legend([h1,h2],'True $\theta$','Posterior mean');
set(hl,'Interpreter','Latex','FontSize',20)
box on

%%
figure(5)
clf
histogram(q(3,:),'Normalization','pdf')
ylimits = get(gca,'YLim');
hold on
h1 = plot([x(2), x(2)],ylimits,'k--','LineWidth',3);
h2 = plot([q_mean(3), q_mean(3)],ylimits,'--','LineWidth',3);
h3 = plot([y(2), y(2)],ylimits,'r--','LineWidth',3);
% plot(xi,f,'Color',[0, 0.4470, 0.7410],'LineWidth',3)
hold off
xlabel('$\theta$','Interpreter','Latex','FontSize',fontsize)
ylabel('$p(x_2|y_{1:T})$','Interpreter','Latex','FontSize',fontsize)
hl = legend([h1,h2, h3],'True $x_2$','Posterior mean','Measured $y_2$');
set(hl,'Interpreter','Latex','FontSize',20)
box on

%%
figure(6)
clf
histogram(q(9,:),'Normalization','pdf')
ylimits = get(gca,'YLim');
hold on
h1 = plot([x(8), x(8)],ylimits,'k--','LineWidth',3);
h2 = plot([q_mean(9), q_mean(9)],ylimits,'--','LineWidth',3);
h3 = plot([y(8), y(8)],ylimits,'r--','LineWidth',3);
% plot(xi,f,'Color',[0, 0.4470, 0.7410],'LineWidth',3)
hold off
xlabel('$\theta$','Interpreter','Latex','FontSize',fontsize)
ylabel('$p(x_8|y_{1:T})$','Interpreter','Latex','FontSize',fontsize)
hl = legend([h1,h2, h3],'True $x_8$','Posterior mean','Measured $y_8$');
set(hl,'Interpreter','Latex','FontSize',20)
box on

%%
figure(2)
clf
subplot 311
histogram(q(1,:))
xlabel('a')
ylabel('p(a | y)')

subplot(3,1,2)
histogram(q(2,:))
xlabel('x_1')
ylabel('p(x_1 | y)')

subplot(3,1,3)
histogram(q(N+1,:))
xlabel('x_N')
ylabel('p(x_N | y)')

figure(3)
plot(q(1,:))


%%
