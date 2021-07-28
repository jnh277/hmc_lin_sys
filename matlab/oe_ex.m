clear all
close all

% I reduced number of data sets for now so it doesnt take as long to run
noDataSets = 10;         % Number of simulations of the true system to run
N = 100;                 % Length of simulation in number of samples
vare = 5*1e-4;          % Measurment noise variance
displaystring = 'Gt';   % Initialise string to use to display results
Gmonte = [];            % In itialize som memory to story estimated Nyquists
close all               % Start from a fresh slate of no figures

base_file_name = 'oe_ex_data';

ip = 'pulse';    % Input can be 'pulse' or 'rand'

for i=1:noDataSets 
 % Set random seed
 % rng(54531445 + i)

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Simulate some data
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 if strcmp(ip,'rand')
  u = randn(1,N);
 elseif strcmp(ip,'pulse')
  u = [zeros(1,floor(0.1*N)),ones(1,floor(0.5*N)),zeros(1,floor(0.4*N))]; u = [u,zeros(1,N-length(u))];
 else
  u = randn(1,N);
 end;
 
 modnum = 2;   % Specify which model to simulate
 
 if modnum==1
  [Btrue, Atrue] = cheby1(3, 5, 0.8);
 elseif modnum==2
  bq = poly([-8.0722,-0.8672,0.0948]);
  aq = real(poly([0.75*exp(j*pi/3),0.75*exp(-j*pi/3),0.95*exp(j*pi/12),0.95*exp(-j*pi/12)]));   
  aq = real(poly([0.8*exp(j*0.50*pi),0.8*exp(-j*0.50*pi)])); 
  bq = bq*sum(aq)/sum(bq); 
  Btrue = [0,bq]; Atrue = aq; 
  
  delta = 0.1;
 % den = real(poly([-0.1,-1,-0.2,-0.3,-0.5,-0.05+j*3,-0.05-j*3]));
  den = real(poly([-10,-9,-1+j*10,-1-j*10]));
  num = den(length(den));
  [bq,aq] = c2dm(num,den,delta,'zoh');
  Btrue = bq; Atrue = aq;

 end;
 
 z = filter(Btrue, Atrue, u);
 noise = sqrt(vare)*randn(size(z));
 y = z + noise;

 Z.y = y; Z.u=u; 
 plot([Z.y(:),z(:)])
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Define and Outpute Error Model Structure
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 M.A     = 9; 
 
%  M.A = length(Atrue)-1+5;  
 
 M.B = 9;
 
 M.type  = 'oe';  M.w = logspace(-3,pi,10000);
 G=est(Z,M);
 
  if i == 1
    Aml = G.A;
    Bml = G.B;
 end
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Estimate on basis of noise corrupted data
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 Gest = est(Z,M); 
 
   if i == 1
    Aml2 = Gest.A;
    Bml2 = Gest.B;
 end
 
 Gest.disp.colour = 'g';
 Gest.disp.linestyle = '--';
 
 Gmonte = [Gmonte,Gest.G(:)];
 

 

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %  Update settings for subsequent display
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 eval(strcat(strcat('G',num2str(i)),'=G;'));
 displaystring = strcat(displaystring,',G',num2str(i));

 %data_estimation = iddata(Z.y(:), Z.u(:));
 %m1 = oe(data_estimation, [4 3 0]);


 file_name = strcat(strcat(base_file_name, int2str(i)), '.mat');
 y_estimation = Z.y; u_estimation = Z.u;
 yhat = filter(G.B,G.A,Z.u);
 coefs_f = G.A;
 coefs_b = G.B;

 A = G.A; B = G.B;


 save(strcat('../data/oe_ex/',file_name), 'y_estimation', 'u_estimation', 'yhat','coefs_b','coefs_f', 'A', 'B','vare');
%  save(file_name, 'y_estimation', 'u_estimation', 'yhat','coefs_b','coefs_f', 'A', 'B','vare');

end  % End loop over noDataSets iterations


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Display the results
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gt.A = Atrue; Gt.B = Btrue; Gt.w = G.w; Gt.type='oe';
Gt = m2f(Gt); Gt.disp.legend = 'True System'; Gt.disp.error=0;

gray = 0.3*ones(1,3);

figure(2)
plot(Gmonte,'Color',gray)
hold on
plot(Gt.G(:),'r','linewidth',4)
title('Prediction error estimates versus truth (bold)')
set(gca,'Fontsize',18);
set(gcf,'Color','white')
%axis([-6,6,-8,8])
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Compute Bayesian Posteriors on the first data set using STAN
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Run stan by "python3 oe_ex.py" at the commend line then when finished'); ...
disp('type "dbcont" in matlab')

keyboard

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Extract Traces
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
load('../results/oe_ex_results.mat');  % Load in data file produced by Stan
% load('results.mat')  % Load in data file produced by Stan

strb = 'b_coefs.'; stra= 'f_coefs.';



% 
for k=1:9 
 Atrace(k,:)=getfield(results,strcat(stra,num2str(k)));
end

for k=1:10 
 Btrace(k,:)=getfield(results,strcat(strb,num2str(k)));
end

for i=1:4
    subplot(2,2,i)
    plot(Atrace(1,200+i:4:800))
    title(['A(1) trace for chain', num2str(i)])
end

% plot bode diagram and compare with HMC estimates
Htrue = tf(Btrue,Atrue,delta);
[MAGtrue,PHASEtrue,W] = bode(Htrue);

Hml = tf(Bml,Aml,delta);
[MAGml,PHASEml] = bode(Hml, W);


sampleMag = nan(length(Atrace),length(W));
samplePhase = nan(length(Atrace),length(W));
for i = 1:length(Atrace)
    Hsamp = tf(Btrace(:,i).',[1., Atrace(:,i).'],delta);
    [MAG,PHASE] = bode(Hsamp, W);
    sampleMag(i,:) = squeeze(MAG);
    samplePhase(i,:) = squeeze(PHASE)-round(max(PHASE)/360)*360;
end


figure()
clf
subplot 211

hold on
% for i = 1:300
s = semilogx(W,20*log10(sampleMag),'g');
for i =1:length(s)
s(i).Color(4) = 0.1;
end
% end
semilogx(W,20*log10(squeeze(MAGtrue)),'k--')
semilogx(W,20*log10(squeeze(MAGml)),'r.-')
hold off
ylabel('Magnitude dB')
xlabel('Frequency rad/s')
title('True = dotted black, hmc samples = green, red=ML')

subplot 212
hold on
s = semilogx(W,samplePhase,'g');
for i =1:length(s)
s(i).Color(4) = 0.1;
end
semilogx(W,squeeze(PHASEtrue),'k--')
semilogx(W,squeeze(PHASEml)-round(max(PHASEml)/360)*360,'r.-')
hold off
ylabel('Phase (deg)')
xlabel('Frequency rad/s')
title('True = dotted black, hmc samples = green, red = ML')


% sigtrace = getfield(results,'sigmae');
% % b_hyp_trace = getfield(results,'b_coefs_hyperprior');
% % a_hyp_trace = getfield(results,'f_coefs_hyperprior');
% 
% energy_trace = getfield(results,'energy__');
% accept_trace = getfield(results,'accept_stat__');
% lp_trace = getfield(results,'lp__');
% divergent_trace = getfield(results,'divergent__');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %
% %  display results
% %
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% binum = 59;
% fignum = 3;
% 
% for k=1:length(Atrue)-1
%  figure(fignum)
%  bc=hist(Atrace(k,:),binum); hist(Atrace(k,:),binum); 
%  hold on
%  plot([Atrue(k+1),Atrue(k+1)],[0,max(bc)],'linewidth',2,'color','r')
%  title(strcat('Parameter A(',num2str(k),')'))
%  hold off
%  fignum=fignum+1;
% end;
% 
% for k=1:length(Btrue)
%  figure(fignum)
%  bc=hist(Btrace(k,:),binum); hist(Btrace(k,:),binum); 
%  hold on
%  plot([Btrue(k),Btrue(k)],[0,max(bc)],'linewidth',2,'color','r')
%  title(strcat('Parameter B(',num2str(k),')')) 
%  hold off
%  fignum=fignum+1;
% end;
% 
% figure(fignum)
%  
% bc=hist(sigtrace,binum); hist(sigtrace,binum); 
% hold on
% plot([sqrt(vare),sqrt(vare)],[0,max(bc)],'linewidth',2,'color','r')
% title('Parameter noise standard deviation') 
% hold off
% fignum=fignum+1;
% 
% Acm = [1,mean(Atrace')];
% Bcm = mean(Btrace');
% 
% %Acm=A; Bcm=B;
% 
% Gcm.A = Acm; Gcm.B = Bcm; Gcm.w = G.w; Gcm.type='oe';
% Gcm = m2f(Gcm); Gcm.disp.legend = 'Conditional Mean'; Gcm.disp.error=0;
% 
% shownyq(Gt,G,Gcm)
% 
% 
% 
% % % Debug to figure out how to do filter in python
% % 
% % ord = len(A);
% % 
% % w(1:length(A))=0.0;
% % 
% % for n=length(A):N
% %  w(n) = 0.0;
% %  for i=1:length(B)
% %   w(n) = w(n) + B(i)*u(n-i+1); 
% %  end;
% %  for i=1:length(A)-1
% %   w(n) = w(n) - A(i+1)*w(n-i)
% %   end;
% % end
% %;
% %
