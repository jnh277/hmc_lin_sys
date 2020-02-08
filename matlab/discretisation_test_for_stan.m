% discretizing state space models

% matrix exponentials can be done in stan using matrix_exp




m = idss(sys);
A = m.A;
B = m.B;
C = m.C;
D = m.D;

Ts = 0.1;

md = c2d(m,Ts);

H = expm([A, B;zeros(size(B)).', zeros(1,1)]*Ts);

Ad = H(1:2,1:2);
Bd = H(1:2,3);
Cd = C;
Dd = D;

Adt = md.A;
Bdt = md.B;

