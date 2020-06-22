function [q_new,p_new,q_hist,p_hist,q_hist_neg,p_hist_neg] = sympleticIntNuts(q,p,y, sig_e, sig_p,M)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
T = 5e-1;
ee = 1e-3;
qq = q;
pp = p;

pp_neg = p;
qq_neg = q;

N = floor(T/ee);
q_hist = nan(N+1,1);
p_hist = nan(N+1,1);
q_hist(1) = q;
p_hist(1) = p;
q_hist_neg = nan(N+1,1);
p_hist_neg = nan(N+1,1);
q_hist_neg(1) = q;
p_hist_neg(1) = p;
for k=1:floor(T/ee)
    [~,glpdf] = model(y, qq, sig_e, sig_p);
    dVdq = - glpdf;
    
    pp = pp - (ee/2)*dVdq;
    
    qq = qq + ee*pp/M;
    
    [~,glpdf] = model(y, qq, sig_e, sig_p);
    dVdq = - glpdf;
    
    pp = pp - (ee/2)*dVdq;
    
    q_hist(k+1) = qq;
    p_hist(k+1) = pp;
    
    %     that was forwards in time, now backwards
    [~,glpdf] = model(y, qq_neg, sig_e, sig_p);
    dVdq = - glpdf;
    pp_neg = pp_neg + (ee/2)*dVdq;
    
    qq_neg = qq_neg - ee*pp_neg/M;
    
    [~,glpdf] = model(y, qq_neg, sig_e, sig_p);
    dVdq = - glpdf;
    pp_neg = pp_neg + (ee/2)*dVdq;
    
    q_hist_neg(k+1) = qq_neg;
    p_hist_neg(k+1) = pp_neg;
    
    d = [qq-qq_neg;pp-pp_neg];
    ddf = [qq - q_hist(k);pp-p_hist(k)];
    ddb = [qq_neg - q_hist_neg(k);pp_neg-p_hist_neg(k)];
    
    if (d.'*ddf < 0) && (d.'*ddb > 0)
        break
    end
    
    %     c1 = pp * (qq-qq_neg);
    %     c2 = pp_neg * (qq_neg - qq);
    %     if (c1 < 0) && (c2 < 0)
    %         break
    %     end
    
    
end

ind = randsample(k+1,1);
% now randomly sample from these trajectories
if rand > 0.5     % sample from forwards
    q_new = q_hist(ind);
    p_new = p_hist(ind);
else              % sample from backwards
    q_new = q_hist_neg(ind);
    p_new = p_hist_neg(ind);
end



end

%%
% figure(10)
% clf
% hold on
% for i=1:length(p_hist)
%     plot(p_hist(1:i),q_hist(1:i),'r','LineWidth',2)
%     plot(p_hist_neg(1:i),q_hist_neg(1:i),'g','LineWidth',2)
%     drawnow
%     pause(0.01)
% end
% hold off

