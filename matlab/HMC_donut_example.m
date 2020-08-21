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
% example of sampling from a donut shaped target using HMC
% given in Section 4 of the paper.

clear all
clc

rng(37) % for repeatability of the plots
paper_plots = true;

sigma = 0.125;
d0 = 1;
target = @(a,b) 1/sqrt(2*pi*sigma^2)*exp(-0.5*(d0 - hypot(a,b)).^2/sigma^2);    % for plotting
log_target = @(theta) -0.5*log(2*pi*sigma^2)-0.5*(d0 - hypot(theta(1,:),theta(2,:))).^2/sigma^2;

[A,B] = meshgrid(linspace(-1.5,1.5));
target_vals = NaN(size(A));
target_vals(:) = target(A(:),B(:));


%%
fontsize = 20;
figure(1)
clf
set(gcf,'Position',[-21   355   682   496])
subplot(3,3,[2 3 5 6 ])
[~, hContour] = contourf(A,B,target_vals,10,'LineStyle','None');
set(gcf, 'Renderer', 'OpenGL');
blue_map = [linspace(1, 0, 256)', linspace(1, 0.4470, 256)',linspace(1, 0.7410, 256)'];
colormap(blue_map)
drawnow;  % this is important, to ensure that FacePrims is ready in the next line!
hFills = hContour.FacePrims;  % array of TriangleStrip objects
[hFills.ColorType] = deal('truecoloralpha');  % default = 'truecolor'
for idx = 1 : numel(hFills)
    hFills(idx).ColorData(4) = 200;
end
hold on
xlim([-1.5 1.5])
ylim([-1.5 1.5])

h_prop = quiver(NaN,NaN,NaN,NaN,'AutoScale','off','Color','k','LineWidth',1);
h_prop.MaxHeadSize = 0.5;
hold off
xlabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\eta_2$','Interpreter','Latex','FontSize',fontsize)


subplot(3,3,[1 4])
marginal1 = trapz(A(1,:),target_vals,2);
marginal1 = marginal1/trapz(B(:,1),marginal1);
% plot(marginal1,A(1,:),'LineWidth',1.5)
% xlim([0 1.5])
plot(A(1,:),marginal1,'LineWidth',1.5)
ylim([0 1.25])
set(gca,'view',[90 -90])
xlabel('$\eta_2$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\pi(\eta_2)$','Interpreter','Latex','FontSize',fontsize)


subplot(3,3,[8 9])
marginal2 = trapz(B(:,1),target_vals,1);
marginal2 = marginal2/trapz(A(1,:),marginal2);
plot(B(:,1),marginal2,'LineWidth',1.5)
ylim([0 1.25])
xlabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\pi(\eta_1)$','Interpreter','Latex','FontSize',fontsize)


if paper_plots
    figure(2)
    clf
    [~, hContour] = contourf(A,B,target_vals,10,'LineStyle','None');
    set(gcf, 'Renderer', 'OpenGL');
    blue_map = [linspace(1, 0, 256)', linspace(1, 0.4470, 256)',linspace(1, 0.7410, 256)'];
    colormap(blue_map)
    drawnow;  % this is important, to ensure that FacePrims is ready in the next line!
    hFills = hContour.FacePrims;  % array of TriangleStrip objects
    [hFills.ColorType] = deal('truecoloralpha');  % default = 'truecolor'
    for idx = 1 : numel(hFills)
        hFills(idx).ColorData(4) = 200; 
    end
    hold on
    h_prop_a = quiver(0,0,0,0,'AutoScale','off','Color','g');
    h_prop_a.MaxHeadSize = 0.8;
    h_prop_a.LineWidth = 1.5;
    
    h_prop_r = quiver(0,0,0,0,'AutoScale','off','Color','r');
    h_prop_r.MaxHeadSize = 0.8;
    h_prop_r.LineWidth = 1.5;
    
  
    hold off
    xlim([-1.5 1.5])
    ylim([-1.5 1.5])
    set(gca,'FontSize',16)
    xlabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize*1.8)
    ylabel('$\eta_2$','Interpreter','Latex','FontSize',fontsize*1.8)

end

%% run hmc
delay = 0.2;
M = 0.01*eye(2);
%  M = [0.5072,   -0.0126;
%    -0.0126,    0.5339];
sM = sqrtm(M);
detM = det(M);
V   = @(q) -log_target(q);
H   = @(q,p) V(q) + 0.5*p.'*(M\p) + log(2*pi*detM);
dVdq = @(q) [-(q(1)*(d0-(q(1)^2 + q(2)^2)^(1/2)))/(sigma^2*(q(1)^2 + q(2)^2)^(1/2));
    -(q(2)*(d0-(q(1)^2 + q(2)^2)^(1/2)))/(sigma^2*(q(1)^2 + q(2)^2)^(1/2))];
%Now we run the main loop
%create space for the chain
num  = 1000;
q   = NaN(2,num);
p   = zeros(2,num);
alp  = zeros(1,num);
err  = zeros(1,num);
% q(:,1) = [0.25;0.25];
q(:,1) = [0.6;0.6];

% prepare some plotting stuff
figure(1)
subplot(3,3,[2 3 5 6 ])
points_line = line(q(1,:),q(2,:),'Color','k','LineStyle','None','Marker','.');
update_line = line(NaN(2,1),NaN(2,1),'Color','k','LineWidth',1.5);

subplot(3,3,7)
htrace = plot(q(1,1),'Marker','.');
xlabel('$k$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize)

if paper_plots
    figure(2)
    points_line2 = line(q(1,:),q(2,:),'Color','k','LineStyle','None','Marker','.','MarkerSize',20);
    hl = legend([hContour,points_line2,h_prop_a,h_prop_r],'$\pi(\eta)$','samples','accepted step','rejected step');
    set([hl],'Interpreter','Latex','FontSize',20,'Location','SouthEast')
    figure(1)
end

acc = 0;
for i=1:num
    %Step 1. propose p
    p(:,i) = sM*randn(2,1);
    %Integrate forwards and backwards in time for some time using leapfrog
    T   = 0.2;
    ee   = 0.005;
    nint  = floor(T/ee);
    qqf  = NaN(2,nint+1);
    ppf  = NaN(2,nint+1);
    qqf(:,1) = q(:,i);
    ppf(:,1) = p(:,i);
    

    for k=1:nint
        tmpf   = ppf(:,k) - (ee/2)*dVdq(qqf(:,k));
        qqf(:,k+1) = qqf(:,k) + ee*(M\tmpf);
        ppf(:,k+1) = tmpf  - (ee/2)*dVdq(qqf(:,k+1));
        
        if i < 30
            update_line.XData = qqf(1,:);
            update_line.YData = qqf(2,:);
            drawnow;
        elseif i==31
            update_line.XData = NaN;
            update_line.YData = NaN;
        end
        %     hnd1=plot(ppf(1:k+1),qqf(1:k+1),'r-x','linewidth',2);
        %     pause(0.1)
        %     drawnow
        %
        
        %     delete(hnd1);
    end
    
    set(h_prop,'Color','k','LineStyle','-')
    h_prop.XData= q(1,i);
    h_prop.YData= q(2,i);
    h_prop.UData= qqf(1,k+1)-q(1,i);
    h_prop.VData= qqf(2,k+1)-q(2,i);

    pause(delay)

    %
    %   delete(hnd1);
    err(i) = H(q(:,i),p(:,i)) - H(qqf(:,k+1),-ppf(:,k+1));
    %Accept or reject and throw away momentum
    alp(i) = min(1,exp(err(i)));
    if rand<alp(i)
        q(:,i+1) = qqf(:,k+1);
        acc = acc + 1;
        set(h_prop,'Color','g') 
        accepted = true;
    else
        q(:,i+1) = q(:,i);
        set(h_prop,'Color','r') 
        accepted = false;
    end
    
    if paper_plots
        if accepted
        h_prop_a.XData= [h_prop_a.XData, q(1,i)];
        h_prop_a.YData= [h_prop_a.YData,q(2,i)];
        h_prop_a.UData= [h_prop_a.UData,qqf(1,k+1)-q(1,i)];
        h_prop_a.VData= [h_prop_a.VData,qqf(2,k+1)-q(2,i)];
        else
        h_prop_r.XData= [h_prop_r.XData, q(1,i)];
        h_prop_r.YData= [h_prop_r.YData,q(2,i)];
        h_prop_r.UData= [h_prop_r.UData,qqf(1,k+1)-q(1,i)];
        h_prop_r.VData= [h_prop_r.VData,qqf(2,k+1)-q(2,i)];    
        end
    end
    
    pause(delay)
    set(h_prop,'Color','k','LineStyle','None')
    
    % update plots
    points_line.XData = q(1,:);
    points_line.YData = q(2,:);
    
    points_line2.XData = q(1,:);
    points_line2.YData = q(2,:);
    if i == 10 || i ==20
        i
    end
    figure(1)
    subplot(3,3,[8,9])
    hold on
    if i > 1
        delete(h2)
    end
    h2 = histogram(q(1,1:i),'BinLimits',[-1.5,1.5],'Normalization','pdf','FaceColor',[0,0.4470,0.7410]);
    
    subplot(3,3,[1,4])
    hold on
    if i>1
        delete(h1)
    end
    h1 = histogram(q(2,1:i),'BinLimits',[-1.5,1.5],'Normalization','pdf','FaceColor',[0,0.4470,0.7410]);
    
    htrace.YData = q(1,1:i+1);
    
    drawnow;
    
end
acc/num