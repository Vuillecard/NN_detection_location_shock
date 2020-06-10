clear all ; 
close all ;
clc ;

alph1=-120*pi/180;
fun1 = @(x,y) 1*((y<tan(alph1))*x)+0*((y>=tan(alph1))*x);

alph2=60*pi/180;
fun2 = @(x,y) 0*(y<tan(alph2)*x)+1*(y>=tan(alph2)*x);

discont_circle = @(x,y) 1*(abs(y)<sqrt(0.5-x.^2))+ 0*( abs(y) >= sqrt(0.5-x.^2));

[X,Y] = meshgrid(-1:.1:1);
Z1 = fun1(X,Y)
Z2 = discont_circle(X,Y)

figure(1)
surf(X,Y,Z1)
figure(2)
surf(X,Y,Z2)