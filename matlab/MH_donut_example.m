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
% example of sampling from a donut shaped target using metropolis hastings
% given in Section 4 of the paper.


rng(37) % for repeatability of plots
paper_plots = true;     % this will plot some figures used in the paper

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

xlabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\eta_2$','Interpreter','Latex','FontSize',fontsize)

h_prop = quiver(NaN,NaN,NaN,NaN,'AutoScale','off','Color','k');
h_prop.MaxHeadSize = 0.5;
hold off


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

%%
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

%% run metropolis hastings
% theta0 = [0,0];
theta0 = [0.6,0.6];
N = 1000;
theta = NaN(2,N);
theta(:,1) = theta0;
sig_mh = [0.3;0.3];
% sig_mh = [0.25;0.25];

figure(1)
subplot(3,3,7)
htrace = plot(theta(1,:),'Marker','.');
xlabel('$k$','Interpreter','Latex','FontSize',fontsize)
ylabel('$\eta_1$','Interpreter','Latex','FontSize',fontsize)

subplot(3,3,[2 3 5 6 ])
points_line = line(theta(1,:),theta(2,:),'Color','k','LineStyle','None','Marker','.');

accept = 0;
delay = 0.1;

if paper_plots
    figure(2)
    points_line2 = line(theta(1,:),theta(2,:),'Color','k','LineStyle','None','Marker','.','MarkerSize',20);
    hl = legend([hContour,points_line2,h_prop_a,h_prop_r],'$\pi(\eta)$','samples','accepted step','rejected step');
    set([hl],'Interpreter','Latex','FontSize',20,'Location','SouthEast')
    figure(1)
end

for i = 1:N-1
    theta_prop = theta(:,i) + sig_mh.*randn(2,1);
    
    a = min(1,exp(log_target(theta_prop)-log_target(theta(:,i))));
    
    % show proposal and whether it gets rejected or not
    set(h_prop,'Color','k')
    h_prop.XData= theta(1,i);
    h_prop.YData= theta(2,i);
    h_prop.UData= theta_prop(1,:)-theta(1,i);
    h_prop.VData= theta_prop(2,:)-theta(2,i);

    pause(delay)
    
    if rand < a     % accept
        theta(:,i+1) = theta_prop;
        accept = accept+1;
        accepted = true;
        set(h_prop,'Color','g') 
    else % reject
        theta(:,i+1) = theta(:,i);
        set(h_prop,'Color','r') 
        accepted = false;
    end
    
    if paper_plots
        if accepted
        h_prop_a.XData= [h_prop_a.XData, theta(1,i)];
        h_prop_a.YData= [h_prop_a.YData,theta(2,i)];
        h_prop_a.UData= [h_prop_a.UData,theta_prop(1,:)-theta(1,i)];
        h_prop_a.VData= [h_prop_a.VData,theta_prop(2,:)-theta(2,i)];
        else
        h_prop_r.XData= [h_prop_r.XData, theta(1,i)];
        h_prop_r.YData= [h_prop_r.YData,theta(2,i)];
        h_prop_r.UData= [h_prop_r.UData,theta_prop(1,:)-theta(1,i)];
        h_prop_r.VData= [h_prop_r.VData,theta_prop(2,:)-theta(2,i)];    
        end
    end
    
    if i == 10 || i == 20 || i == 30 || i == 40
       i 
    end
    
    pause(delay)

    
    %% update all the plots
    points_line.XData = theta(1,:);
    points_line.YData = theta(2,:);
    
    points_line2.XData = theta(1,:);
    points_line2.YData = theta(2,:);
     
    figure(1)
    subplot(3,3,[8,9])
    hold on
    if i > 1
        delete(h2)
    end
    h2 = histogram(theta(1,1:i),'BinLimits',[-1.5,1.5],'Normalization','pdf','FaceColor',[0,0.4470,0.7410]);

    subplot(3,3,[1,4])
    hold on
    if i > 1
        delete(h1)
    end
    h1 = histogram(theta(2,1:i),'BinLimits',[-1.5,1.5],'Normalization','pdf','FaceColor',[0,0.4470,0.7410]);
    
    htrace.YData = theta(1,:);
    
    drawnow;
    
    
    
    
    
end

display(['acceptance ratio = ', num2str(accept/N)])

%%


