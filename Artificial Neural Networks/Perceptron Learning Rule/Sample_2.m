clc;clear;close all;

% Input example => [x1 x2 target]
x = [0.1 0.2 1
    -0.1 -0.3 0
    -0.6 -0.6 0
    0.2 0.4 1
    0.5 0.7 1
    -0.1 -0.7 0
    0.3 0.3 1
    0.8 0.3 1
    -0.3 -0.4 0
    -0.5 -0.2 0];
sizeX = size(x);

% Config Values
w = rand(1,2);
w = 2*w - 1;
theta = rand();
theta = 2*theta - 1;
eta = 0.1;
flag = [0 0 0 0 0 0 0 0 0 0];
epoch = 1 ;

% Plot
xlim([-1.5 1.5])
ylim([-1.5 1.5])
hold on
grid on
for i=1:sizeX(1,1)
    if x(i,3) == 1
        plot(x(i,1),x(i,2),'oc','LineWidth',5)
    else
        plot(x(i,1),x(i,2),'or','LineWidth',5)
    end
end

% Training
while ~(flag(1) && flag(2) && flag(3) && flag(4) && flag(5) && flag(6) && flag(7) && flag(8) && flag(9) && flag(10))
    disp(['Marhale: ',num2str(epoch)])
    for i=1:sizeX(1,1)
        net = x(i,1)*w(1) + x(i,2)*w(2);
        
        % Step Function
        if net >= theta
            output = 1;
        end
        if net < theta
            output = 0;
        end
        
        w(1) = w(1) + eta*(x(i,3) - output)*x(i,1);
        w(2) = w(2) + eta*(x(i,3) - output)*x(i,2);
        theta = theta + eta*(x(i,3) - output)*(-1);

        if x(i,3) - output == 0
            flag(i) = true;
        else
            flag(i) = false;
        end

        T = table(x(i,1),x(i,2),theta,w(1),w(2),output,x(i,3),'VariableNames',{'X1','X2','THETA','W1','W2','OUTPUT','TARGET'}); 
        disp(T)

        xP = -1.5 : 0.1 : 1.5 ;
        yP = (-w(1) ./ (w(2)))*xP + (theta ./ w(2)) ;
        plot(xP,yP,'b','LineWidth',1)

        pause(0.1)
    end
    epoch = epoch + 1;
end

plot(xP,yP,'k','LineWidth',2)