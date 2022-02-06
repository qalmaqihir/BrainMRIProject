function some_plots()
x= [0:0.01:10];
y=sin(x);

subplot(1,2,1)
plot(x,y)
xlabel('x')
ylabel('Sine(x)')
title('Sin(x) Graph')
grid on, axis  equal

z=cos(x);
subplot(1,2,2)
plot(x,z)

