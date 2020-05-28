% time steps
N = 500;

% horizon
M = 50;

trajectory = repmat([10;10;0;0;0],1,N+M+1);

% Mobile robot model parameters
param = buildMobileRobotModel();        % the true parameters

%% the estimated params
hmc_param = load('./rover_parameter_traces.mat');
m = hmc_param.mass;
l = hmc_param.length;
J = hmc_param.inertia;

M_upper_CI = prctile(hmc_param.mass,97.5);
M_lower_CI = prctile(hmc_param.mass,2.5);

l_upper_CI = prctile(hmc_param.length,97.5);
l_lower_CI = prctile(hmc_param.length,2.5);

J_upper_CI = prctile(hmc_param.inertia,97.5);
J_lower_CI = prctile(hmc_param.inertia,2.5);

n_sub = 200;
inds = randsample(length(m),n_sub);

m_hmc = m(inds);
l_hmc = l(inds);
J_hmc = J(inds);

inds_outside = logical((m(inds) < M_lower_CI) + (m(inds) > M_upper_CI) +...
    (l(inds) < l_lower_CI) + (l(inds) > l_upper_CI) + ...
    (J(inds) < J_lower_CI) + (J(inds) > J_upper_CI));
% 
while(any(inds_outside))
    N_new = sum(inds_outside);
    inds_new = randsample(length(m),N_new);
    inds(inds_outside) = inds_new;
    inds_outside = logical((m(inds) < M_lower_CI) + (m(inds) > M_upper_CI) +...
        (l(inds) < l_lower_CI) + (l(inds) > l_upper_CI) + ...
        (J(inds) < J_lower_CI) + (J(inds) > J_upper_CI));
    
end



param.m_hmc = m(inds);
param.l_hmc = l(inds);
param.J_hmc = J(inds);



%%

%Simulate the robot
x = zeros(5,N+1);

% inputs and how far to look ahead

U = zeros(2,M);

% Trajectory just needs to contain column of x, y, h, v, w

for t=1:N
    %Solve MPC problem for input sequence
    U      = solveMPCconstrained(x(:,t),[U(:,2:end) U(:,end)],trajectory(:,t:t+M-1),param);
%     U      = solveMPCconstrained_hmc(x(:,t),[U(:,2:end) U(:,end)],trajectory(:,t:t+M-1),param);

    u(:,t) = U(:,1);
    
    [tn,xn]  = mobileRobotStateSpace(0.1*[(t-1):t],x(:,t),u(:,t),param);
    x(:,t+1) = xn(end,:).';
    
    
    pose.x = x(1,t);
    pose.y = x(2,t);
    pose.h = x(3,t);
    t
    
%     [scan.range(:,t), scan.x(:,t), scan.y(:,t), scan.rexact(:,t), scan.index(:,t)] = getScanPlusNoise(pose,map,laser);
    

end
%%
theta = linspace(0,2*pi);
r = 2.5;
xc = r*cos(theta) + 5;
yc = r*sin(theta) + 5;

% xc2 = r*cos(theta) + 11;
% yc2 = r*sin(theta) + 5;

%%1
figure(1)
clf
plot(x(1,:),x(2,:),'LineWidth',2)
hold on
plot(xc,yc,'k')
% plot(xc2,yc2,'k')
hold off
axis equal
