function [dH, H] = hamiltonian(t, z, y, sig_e, sig_p, M)
%UNTITLED12 Summary of this function goes here
%   Detailed explanation goes here
q = z(1);
p = z(2);


% first the potential part that comes from pi(q)
[lpdf,glpdf] = model(y, q, sig_e, sig_p);
V = -lpdf;
gV = -glpdf;

% now the kinetic part that comes from 
[lpdf, glpdf] = logNormal(p, 0, sqrt(M));
K = -lpdf;
gK = -glpdf;

H = V + K;
dH = [gK;
     -gV];
    


end

