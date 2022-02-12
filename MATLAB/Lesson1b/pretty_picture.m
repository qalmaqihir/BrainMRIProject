
function pretty_picture(N)

t = 0:(.99*pi/2):N;
x =t.*cos(t);
%x=t.*sin(t);
%x = t.*tan(t);
%y =t.*sin(t);
y=t.*sin(t);
plot(x,y,'k')
axis square

