%%

x=0:0.05:5;
y=sin(x.^2);
figure, plot(x,y)
grid
xlabel('Degree')
ylabel("sin(x^2)")

title("Graph of sine^2")


%%
x = [0:5:100];
y = x;
plot(x,y);
grid
%%
x = [-100:20:100];
x = [-10:10];
y = x.^2;
plot(x,y);
xlabel('Values');
ylabel('Sqaures');
grid on;
axis equal;
%%
x=[0:0.01:10];
y=sin(x);
figure,plot(x,y);
xlabel("Values of x");
ylabel('Sin(x)');
title("Graph of sin(x)");
grid on;
axis square;

%%
x = 0:pi/100:2*pi;
y1 = 2*cos(x);
y2 = cos(x);
y3 = 0.5*cos(x);
plot(x,y1, '--',x,y2, "-",x,y3, ":")
legend('2*cos(x)', 'cos(x)', '0.5*cos(x)')
grid
%%

x = [0:0.01:10];
y = sin(x);
g = cos(x);
plot(x,y,x,g,"--")
grid
legend('sin(x)', 'cos(x)')
%%
x=0:.1:2*pi;
y=sin(x);
plot(x,y);
grid on;
hold on;
plot(x,exp(-x),'k:o');
hold off;
axis([0 2*pi 0 1])

title('2-D plot')
xlabel('Time');
ylabel('F(t)');
text(pi/3, sin(pi/3),'<---- sin(pi/3)');
legend('Sin Wave', 'Decaying Exponential');

%% 
x = pi*(-1:.01:1);
y = sin(x);
clf;
plot(x,y,"r-");
grid
%%
c = -2.9:0.2:2.9;
x = randn(5000,1);
hist(x,c);
grid



%%
x=[0:0.5:5];
y=xp(-1.5*x).*sin(10*x);
subplot(1,2,1);
%%
t=0:pi/10:2*pi;
[X,Y,Z]=cylinder(4*cos(t));
subplot(221)
mesh(X)
subplot(222)
mesh(Y)
subplot(223)
mesh(Z)
subplot(224)
mesh(X,Y,Z)

figure,plot3(X,Y,Z)
%%
x = [1:10];
y = [75,58, 90,87,50,85,92,75,60,95];
figure,
bar(x,y)
xlabel("Student")
ylabel("Score")
title("First sem:")

%%
t=0:pi/50:10*pi;
x=sin(t);
y=cos(t);
z=t;
h=plot3(x,y,z,'g-');
set(h,'LineWidth',4*get(h,'LineWidth'));
grid
%%
t=0:pi/50:*pi;
x=sin(t);
y=cos(t);
z=t;
subplot(221)
fill3(x,y,z,'g')
subplot(222)
stem3(x,y,z,'r')
subplot(223)
h=plot3(x,y,z,'g-');
set(h,'LineWidth',4*get(h,'LineWidth'));
subplot(224)
mesh(x,y,z)

% set(h,'LineWidth',4*get(h,'LineWidth'));
grid on

%%
















