function [f,g] = logBarrierRadial(rs,x0,y0,c0,x,y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

rsp = sqrt((x-x0).^2+(y-y0).^2);
inds = rsp <= rs*1.01;
f = nan(size(x));
f(~inds) = -log(c0*(-rs+rsp(~inds)));
f(inds) = 10;

inds2 = rsp >= (c0*rs+1)/c0;
f(inds2) = 0;
% inds = rsp >= rs;
% f(inds) = inf;

if nargout >= 2
    g = nan(2,length(x(:)));
    g(1,:) = -(x-x0)./(rsp.*(rsp-rs));
    g(1,inds) = 0;
    g(2,:) = -(y-y0)./(rsp.*(rsp-rs));
    g(2,inds) = 0;
end
end

