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


model='new';
% model='new';
model = 'v3';

max_samples = 4000;


%% load in data
if strcmp(model,'old')
    % define simulation model
    nx = 6;
    n=6; p=1; m=1;
    den = real( poly([-0.1,-0.2,-0.02+j*1,-0.02-j*1,-0.01-j*0.1,-0.01+j*0.1]) );
    num = 10*den(length(den));
    [a,b,c,d] =tf2ss(num,den); 
    sysc=ss(a,b,c,d); 
    
    % load hmc results
    hmc_sysid_results = load("../results/ctrl_example_sysid");
    
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

elseif strcmp(model,'new')
    % define simulation model
    delta = 0.1;
    den = real(poly([-10,-9,-1+j*10,-1-j*10]));
    num = den(length(den));
    [bq,aq] = c2dm(num,den,delta,'zoh');
    sys_true = tf(bq,aq,delta);
    
%     [a,b,c,d] =tf2ss(num,den); 
%     sysc=ss(a,b,c,d);
    
    % load hmc results
    hmc_sysid_results = load("../results/example2_samples.mat");
    
    ind = randsample(length(hmc_sysid_results.tf_num),max_samples);
    num_hmc = hmc_sysid_results.tf_num(ind,:);
    den_hmc  = hmc_sysid_results.tf_den(ind,:);
    
    % add ones to front of den as hmc is not estimating this
    den_hmc = [ones(length(ind),1), den_hmc];
    
    num_mean = mean(num_hmc,1);
    den_mean = mean(den_hmc,1);
    
    sys_cm_tf = tf(num_mean,den_mean,delta);
    sys_cm_ss = ss(sys_cm_tf);
%     syscm = ss((tf(num_mean,den_mean,delta)));
    
    nx = length(syscm.A);
elseif strcmp(model,'v3')
    % define simulation model
    nx = 6;
    n=6; p=1; m=1;
    den = real( poly([-0.1,-0.2,-0.02+j*1,-0.02-j*1,-0.01-j*0.1,-0.01+j*0.1]) );
    num = 10*den(length(den));
    [a,b,c,d] =tf2ss(num,den); 
    sysc=ss(a,b,c,d); 
    delta = 1;
    sys_true = c2d(sysc, delta);
    
        % load hmc results
    hmc_sysid_results = load("../results/example2_samples2.mat");
    ind = randsample(length(hmc_sysid_results.tf_num),max_samples);
    num_hmc = hmc_sysid_results.tf_num(ind,:);
    den_hmc  = hmc_sysid_results.tf_den(ind,:);
    
    % add ones to front of den as hmc is not estimating this
    den_hmc = [ones(length(ind),1), den_hmc];
    
    num_mean = mean(num_hmc,1);
    den_mean = mean(den_hmc,1);
    
    sys_cm_tf = tf(num_mean,den_mean,delta);
    sys_cm_ss = ss(sys_cm_tf);
    
    nx = length(num_mean)-1;
end


%%

%% first work out the conditional mean system
% Now iterate through all parameter realisations

% [MAGml,PHASEml] = bode(Hml, W);

figure(1)
clf
% bode(sysc
bode(sys_true)
hold on
bode(sys_cm_ss)
hold off
% xlim([1e-1,3e1])

%% phase and gain margins for slow controller

%These are controller weighting that remain constant for all controller
%designs
if strcmp(model,'old')
    Qx = 1*eye(nx);   %State penalty
    Ru = 1;        %Input penalty
    Qw = eye(nx);   %State noise covariance
    Rv = 10;        %Measurement noise covariance
    Qi = 0.01;        %Penalty on integral term of e=r-y
elseif strcmp(model,'new')
    Qx = 100*eye(nx);   %State penalty
    Ru = 0.01;        %Input penalty
    Qw = 0*eye(nx);   %State noise covariance
    Rv = 0.05^2;
    Qi = 1e7;        %Penalty on integral term of e=r-y
elseif strcmp(model,'v3')
%     Qx = 1*eye(nx);   %State penalty
%     Ru = 1;        %Input penalty
%     Qw = eye(nx);   %State noise covariance
%     Rv = 15;        %Measurement noise covariance
%     Qi = 0.01;        %Penalty on integral term of e=r-y
    
    Qx = 1*eye(nx);   %State penalty
    Ru = 1;        %Input penalty
    Qw = eye(nx);   %State noise covariance
    Rv = 500;        %Measurement noise covariance
    Qi = 0.001;        %Penalty on integral term of e=r-y
    
end

% Specify the controller to be used
n=nx;

% syscm = conditional mean system

% Kslow = lqg(ss(syscm),blkdiag(Qx,Ru),blkdiag(Qw,Rv),Qi);
Kslow = lqg(sys_cm_ss,blkdiag(Qx,Ru),blkdiag(Qw,Rv),Qi);

Tsys_cm   = Kslow*sys_cm_ss;
[gm,pm] = margin(-Tsys_cm(1,2));
nom_slow = pm;
% Tsys = Kslow*ss(syscm)
fslow = zeros(3,max_samples);
%
% Now iterate through all parameter realisations
parfor k=1:max_samples
    k
    if strcmp(model,'old')
        sys=ss(tf(num_hmc(k,:),den_hmc(k,:)));
    elseif strcmp(model,'new')
        sys_tf = tf(num_hmc(k,:),den_hmc(k,:),delta);
        sys = ss(sys_tf);
    elseif strcmp(model,'v3')
        sys_tf = tf(num_hmc(k,:),den_hmc(k,:),delta);
        sys = ss(sys_tf);
    end
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


% v3
if strcmp(model,'old')
    Qx = 1*eye(nx);   %State penalty
    Ru = 0.01;        %Input penalty
    Qw = eye(nx);   %State noise covariance
    Rv = 0.4;        %Measurement noise covariance
    Qi = 100;        %Penalty on integral term of e=r-y
elseif strcmp(model,'new')
    Qx = 100*eye(nx);   %State penalty
    Ru = 0.0001;        %Input penalty
    Qw = 0*eye(nx);   %State noise covariance
%     Rv = 10;        %Measurement noise covariance
    Rv = 0.05^2;
    Qi = 1e8;        %Penalty on integral term of e=r-y
elseif strcmp(model,'v3')
    Qx = 10*eye(nx);   %State penalty
    Ru = 0.01;        %Input penalty
    Qw = eye(nx);   %State noise covariance
    Rv = 10;        %Measurement noise covariance
    Qi = 0.1;        %Penalty on integral term of e=r-y
end

% Specify the controller to be used
n=nx;
Kfast = lqg(sys_cm_ss,blkdiag(Qx,Ru),blkdiag(Qw,Rv),Qi);

Tsys_cm   = Kfast*sys_cm_ss;
[gm,pm] = margin(-Tsys_cm(1,2));
nom_fast = pm;

% Now iterate through all parameter realisations
parfor k=1:max_samples
    k
    if strcmp(model,'old')
        sys=ss(tf(num_hmc(k,:),den_hmc(k,:)));
    elseif strcmp(model,'new')
        sys_tf = tf(num_hmc(k,:),den_hmc(k,:),delta);
        sys = ss(sys_tf);
    elseif strcmp(model,'v3')
        sys_tf = tf(num_hmc(k,:),den_hmc(k,:),delta);
        sys = ss(sys_tf);
    end
    Tsys   = Kfast*sys;
    [gm,pm] = margin(-Tsys(1,2))
    st = nan;
    ffast(:,k) = [gm;pm;st];
end;

figure(5)
clf
hist(ffast(2,:))


%%
% a little bit of filtering
% ffast(:,ffast(2,:)<-30) = [];
% ffast(:,ffast(2,:)>90) = [];

XI = linspace(-40,50,500);

fontsize = 22;

figure(2)
[F,XI] = ksdensity(fslow(2,:), XI);
plot(XI,F,'LineWidth',3)
hold on
[F,XI] = ksdensity(ffast(2,:), XI);
plot(XI,F,'--','LineWidth',3)
set(gca,'FontSize',16)
ylims = get(gca,'YLim');
plot([0 0],ylims,'k--')
ylim(ylims)
% histogram(ffast(2,:),'Normalization','pdf')
hold off
hl = legend('Slow controller','Fast controller');
set(hl,'FontSize',fontsize,'Interpreter','Latex')
% xlim([-15 35])
xlabel('Phase margin (degrees)','FontSize',fontsize,'Interpreter','Latex')
ylabel('PDF','FontSize',fontsize,'Interpreter','Latex')
grid on

%%
