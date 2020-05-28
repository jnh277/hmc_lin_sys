% test log cost


x = linspace(0,1,1000);
b = 0;
c0 = 100;
f = logBarrier(b,c0,x);

figure(1)
clf
plot(x,f)

%% radial log barrier


[X,Y] = meshgrid(linspace(-1,1,1000));

rs = 0.2;
x0 = 0.1;
y0 = 0;
c0 = 100;


[f] = real(logBarrierRadial(rs,x0,y0,c0,X,Y));

figure(2)
clf
pcolor(X,Y,f)
shading interp

%% raidal log barrier graidnet check
x = linspace(0,1,1000);
[f,g] = logBarrierRadial(rs,x0,y0,c0,x,0);

figure(4)
plot(x,f)

figure(3)
clf
plot(x,gradient(real(f),x(2)-x(1)))
hold on
plot(x,g,'--')
hold off

