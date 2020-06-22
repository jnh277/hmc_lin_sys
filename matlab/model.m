function [lpdf,glpdf] = model(y, theta, sig_e, sig_p)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

N = length(y);

% prior
[lpdf, glpdf] = logNormal(theta, 0, sig_p);

% likelihood
[tmp1, tmp2] = logNormal(y(1:N-1)*theta, y(2:N), sig_e);
lpdf = lpdf + tmp1;
glpdf = glpdf + y(1:N-1).'*tmp2;



end

