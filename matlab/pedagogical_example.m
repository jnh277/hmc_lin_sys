
%% measurement parameters
T = 20;         % number
y0 = 1;         % initial
sig_e = 0.2;    % noise
a = 0.9;


%% inference parameter
Sigma_p = 1;    % prior variance of a

%% hmc parameters

autotune_mass = false;     
M = 0.2;         % init mass matrix to use when autotuning
N = 400;        % iterations
NS = 5;        % number of samples to grab per iteration
warmup = 20;
q0 = 0;         % initialise q

%% simulate measurements
y = zeros(T,1);
y(1) = y0;

for i=1:T-1
   y(i+1) = a*y(i) + sig_e*randn(1);
    
    
end

figure(1)
clf
plot(y)
xlabel('t')
ylabel('y')

%% analytical result
Phi = y(1:T-1);
m = 1;
Gamma = [Phi./(sig_e*ones(T-1,1));diag(1./sqrt(Sigma_p))];
R = triu(qr(Gamma));
C = R(1:m,1:m);
Q = inv(C);
a_post = (C\(C'\(Phi'*y(2:T)/sig_e^2)));
a_var = Q*Q.';

%% produce energy contours
q_grid = linspace(0,2,150);
p_grid = linspace(-2,2);
H_grid = zeros(length(q_grid),length(p_grid));
for i = 1:length(q_grid)
    for j = 1:length(p_grid)
        z = [q_grid(i);p_grid(j)];
        [~, H_grid(i,j)] = hamiltonian(0, z, y, sig_e, sig_p, M);
        
        
    end
end

figure(4)
clf
contour(p_grid,q_grid,H_grid)
ylabel('q - generalised coordinate')
xlabel('p - momentum')
title('Energy levels')
axis square

%%

q = nan(N,1);
q(1) = q0;
tmax = 0.05;
count = 0;
for i = 1:N-1
   %% draw a p
   p = sqrt(M)*randn(1); 
   
   % compute the hamiltonian at this start spot
   z0 = [q(i);p];
   [~, H0] = hamiltonian(0, z0, y, sig_e, sig_p, M);
   
%    % integrate to find a new spot
%    hfunc = @(t,z) hamiltonian(t,z,y, sig_e, sig_p, M);
%    % integrate forwards
%    [t,z] = ode45(hfunc, linspace(0,tmax), z0);
%    % integrate backwards
%    ind = randsample(100,1);
%    zl = [z(ind(1),1), -z(ind(1),2)];

    % try simpletic integrator
%     [q_new,p_new,q_hist,p_hist] = sympleticInt(q(i),p,y, sig_e, sig_p,M);
    [q_new,p_new,q_hist,p_hist,q_hist_neg,p_hist_neg] = sympleticIntNuts(q(i),p,y, sig_e, sig_p,M);
    zl = [q_new;p_new];

   [~, Hl] = hamiltonian(0, zl, y, sig_e, sig_p, M);
   
   % now choose whether to accept or reject
   alpha = min(1, exp(-Hl + H0));
   
   if rand > alpha % use old
      q(i+1) = q(i); 
       
   else     % use new
      q(i+1) = zl(1);
       count = count+1;
   end
   
   if autotune_mass
       if ~mod(i,100) && i > warmup
           M = var(q(20:i));
       end
   end
   
    figure(3)
    plot(q);
    hold on
    plot([1,i],[1 1]*a_post,'--')
    plot([1,i],[1 1]*a_post+2*sqrt(a_var),'g--')
    plot([1,i],[1 1]*a_post-2*sqrt(a_var),'g--')
    hold off
    legend('trace','posterior mean','95% CI')
    drawnow
    
    
    if i > 20
        figure(4)
        hold on
        plot(p_hist,q_hist,'r')
        plot(p_hist_neg,q_hist_neg,'g')
        hold off
    
    end
%     
%     title(['M = ', num2str(M)])

end







%%
acceptance_ratio = count/N;

q = q(warmup:end);

post_analytic = normpdf(a_grid,a_post,sqrt(a_var));

figure(2)
clf
plot(a_grid,post_analytic)
hold on
histogram(q,'Normalization','pdf')
ylabel('p(a | y)')
xlabel('a')

%%






