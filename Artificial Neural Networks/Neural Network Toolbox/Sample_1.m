close all
clear
clc

% Function
x=-6:0.05:6;
t=sin(x)+x.*cos(3*x);

% Toolbox training
nn=20;
net = fitnet(nn);
net.trainParam.epochs=1000;
net.Divideparam.trainRatio=0.7;
net.Divideparam.valRatio=0.2;
net.Divideparam.testRatio=0.1;
net = train(net,x,t);

% Output
x2=-6:0.01:6;
t2=net(x2);

% Plot
hold on
plot(x,t,'o');
plot(x2,t2,'r');