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

% This script uses the system identification resutls from example6_sysid.py
% and example6_plotsysid.py to design two controllers a fast and a slow
% controller that show the utility of having samples from the full
% posterior distribution. It produces the controll results plots shown in
% the paper for example 6 (section 6.7)
%% load in data
nx = 6;
n=6; p=1; m=1;
den = real( poly([-0.1,-0.2,-0.02+j*1,-0.02-j*1,-0.01-j*0.1,-0.01+j*0.1]) );
num = 10*den(length(den));
[a,b,c,d] =tf2ss(num,den); 
sysc=ss(a,b,c,d); 

hmc_sysid_results = load("../results/ctrl_example_sysid");
%%
max_samples = 4000;
ind = randsample(length(hmc_sysid_results.A_traces),max_samples);

A_hmc = hmc_sysid_results.A_traces(ind,:,:);
B_hmc = hmc_sysid_results.B_traces(ind,:);
C_hmc = hmc_sysid_results.C_traces(ind,:);
D_hmc = hmc_sysid_results.C_traces(ind);
num_hmc = hmc_sysid_results.tf_nums(ind,:);
den_hmc  = hmc_sysid_results.tf_dens(ind,:);
num_mean = hmc_sysid_results.tf_num_mean;
den_mean = hmc_sysid_results.tf_den_mean;

syscm = ss(tf(num_mean,den_mean));
%% first work out the conditional mean system
% Now iterate through all parameter realisations

figure(1)
clf
bode(sysc)
hold on
bode(syscm)
hold off


%% phase and gain margins for slow controller




%These are controller weighting that remain constant for all controller
%designs
Qx = eye(nx);   %State penalty
Ru = 1;        %Input penalty
Qw = eye(nx);   %State noise covariance
Rv = 10;        %Measurement noise covariance
Qi = 0.01;        %Penalty on integral term of e=r-y

% Specify the controller to be used
n=nx;

% syscm = conditional mean system

Kslow = lqg(ss(syscm),blkdiag(Qx,Ru),blkdiag(Qw,Rv),Qi);


% Tsys = Kslow*ss(syscm)
fslow = zeros(3,max_samples);
%%
% Now iterate through all parameter realisations
parfor k=1:max_samples
    k
    sys=ss(tf(num_hmc(k,:),den_hmc(k,:)));
    Tsys   = Kslow*sys;
%     s      = allmargin(-Tsys(1,2));
%     gm     = min(s.GainMargin);
%     if isempty(gm), gm=nan; end
%     pm     = min(s.PhaseMargin(s.PhaseMargin>0));
%     if isempty(pm), pm=nan; end
%     st     = s.Stable;
%     if length(st)>1, st=nan; end
    [gm,pm] = margin(-Tsys(1,2))
    st = nan;
    fslow(:,k) = [gm;pm;st];
end;


%% fast controller
% Reserve some RAM to store function of these parameter realisations
ffast = zeros(3,max_samples);

%These are controller weighting that remain constant for all controller
%designs

% old vals
% Qx = eye(nx);   %State penalty
% Ru = 1;        %Input penalty
% Qw = eye(nx);   %State noise covariance
% Rv = 0.4;        %Measurement noise covariance
% Qi = 0.002;        %Penalty on integral term of e=r-y

% Qx = 1*eye(nx);   %State penalty
% Ru = 0.01;        %Input penalty
% Qw = eye(nx);   %State noise covariance
% Rv = 0.4;        %Measurement noise covariance
% Qi = 10;        %Penalty on integral term of e=r-y

% v3
Qx = 1*eye(nx);   %State penalty
Ru = 0.01;        %Input penalty
Qw = eye(nx);   %State noise covariance
Rv = 0.4;        %Measurement noise covariance
Qi = 100;        %Penalty on integral term of e=r-y

% Specify the controller to be used
n=nx;
Kfast = lqg(syscm,blkdiag(Qx,Ru),blkdiag(Qw,Rv),Qi);

% Now iterate through all parameter realisations
parfor k=1:max_samples
    k
    sys=ss(tf(num_hmc(k,:),den_hmc(k,:)));
    Tsys   = Kfast*sys;
%     s      = allmargin(-Tsys(1,2));
%     gm     = min(s.GainMargin);
%     if isempty(gm), gm=nan; end
%     pm     = min(s.PhaseMargin(s.PhaseMargin>0));
%     if isempty(pm), pm=nan; end
%     st     = s.Stable;
%     if length(st)>1, st=nan; end
    [gm,pm] = margin(-Tsys(1,2))
    st = nan;
    ffast(:,k) = [gm;pm;st];
end;

%%
% a little bit of filtering
ffast(:,ffast(2,:)<-30) = [];
ffast(:,ffast(2,:)>90) = [];

fontsize = 22;

figure(2)
[F,XI] = ksdensity(fslow(2,:));
plot(XI,F,'LineWidth',3)
hold on
[F,XI] = ksdensity(ffast(2,:));
plot(XI,F,'--','LineWidth',3)
set(gca,'FontSize',16)
ylims = get(gca,'YLim');
plot([0 0],ylims,'k--')
ylim(ylims)
% histogram(ffast(2,:),'Normalization','pdf')
hold off
hl = legend('Slow controller','Fast controller');
set(hl,'FontSize',fontsize,'Interpreter','Latex')
xlim([-15 35])
xlabel('Phase margin (degrees)','FontSize',fontsize,'Interpreter','Latex')
ylabel('PDF','FontSize',fontsize,'Interpreter','Latex')
grid on

%%
% Kslow=minreal(tf(0.005,1,1) + tf([0.001,0],[1,-1],1));
% Kfast=minreal(tf(0.02,1,1) + tf([0.002,0],[1,-1],1));
% figure(1)
% step(((Kslow*c2d(syscm,1))/(1+Kslow*c2d(syscm,1)))); hold on; 
% figure(2)
% step(((Kfast*c2d(syscm,1))/(1+Kfast*c2d(syscm,1)))); hold off;
% 
% %%
% ffast = zeros(3,max_samples);
% fslow = zeros(3,max_samples);
% parfor k = 1:max_samples
%     sys_s = ss(tf(num_hmc(k,:),den_hmc(k,:)));
%     s = allmargin(((Kfast*c2d(sys_s,1))/(1+Kfast*c2d(sys_s,1))));
%     gm     = min(s.GainMargin);
%     if isempty(gm), gm=nan; end
%     pm     = min(s.PhaseMargin(s.PhaseMargin>0));
%     if isempty(pm), pm=nan; end
%     st     = s.Stable;
%     if length(st)>1, st=nan; end
%     ffast(:,k) = [gm;pm;st];
%     
%     s = allmargin(((Kslow*c2d(sys_s,1))/(1+Kslow*c2d(sys_s,1))));
%     gm     = min(s.GainMargin);
%     if isempty(gm), gm=nan; end
%     pm     = min(s.PhaseMargin(s.PhaseMargin>0));
%     if isempty(pm), pm=nan; end
%     st     = s.Stable;
%     if length(st)>1, st=nan; end
%     fslow(:,k) = [gm;pm;st];
%     k
% end
% 
% % figure(4)
% % clf
% % histogram(ffast(1,:),'Normalization','pdf')
% % hold on
% % histogram(fslow(1,:),'Normalization','pdf')
% % hold off
% % xlim([0,1])
%     
%     