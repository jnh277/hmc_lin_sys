% explroe prior regularisation impacts for ssm models


N = 200;
eigs = nan(4,N);

figure(2)
clf
hold on

for i = 1:N

% A = 5*randn(4,4); A = A.'*A;
A = 40*randn(4,4);
et = real(eig(A));
if all(et>0)
    eigs(:,i) = et;
    B = randn(4,1);
    C = randn(1,4);
    D = 0;
    bode(A,B,C,D)
end




end
hold off
title('bode plots when systems have std=1 prior')

figure(1)
clf
histogram(eigs(:))
title('Eigenvalues when values have std=1 prior')
