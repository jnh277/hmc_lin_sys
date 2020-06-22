function [q_new,p_new,q_hist,p_hist] = sympleticInt(q,p,y, sig_e, sig_p,M)
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here
  T = 0.5e-1;
  ee = 1e-3;
  qq = q;
  pp = p;
  N = floor(T/ee);
  q_hist = nan(N+1,1);
  p_hist = nan(N+1,1);
  q_hist(1) = q;
  p_hist(1) = p;
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
    
%     d = [q-qq;p-pp];
%     dd = [qq-q_hist(k);pp-p_hist(k)];
%     if d.'*dd > 0
%         kk = floor((k+1)/2);
%         qq = q_hist(kk);
%         pp = p_hist(kk);
%         q_hist = q_hist(1:kk);
%         p_hist = p_hist(1:kk);
%         break
%     end
    
  end
  
  % randomly sample this trajectory
  ind = randsample(k+1,1);
  q_new = q_hist(ind);
  p_new = p_hist(ind);
  % trim trajectories for plotting
  q_hist = q_hist(1:ind);
  p_hist = p_hist(1:ind);
end

