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
sizeX = size(x, 1);

% Rescale
temp = rescale(x(:,1:2));
temp(:,3) = x(:,3);
x = temp;

% Config Values
w = rand(1,2);
theta = rand();
eta = 0.5;
epoch = 1 ;
epsilon = 0.01;
oldError = 0;

% Plot
xlim([-0.5 1.5])
ylim([-0.5 1.5])
hold on
grid on
for i=1:sizeX
    if x(i,3) == 1
        plot(x(i,1),x(i,2),'oc','LineWidth',5)
    else
        plot(x(i,1),x(i,2),'or','LineWidth',5)
    end
end

% Training
while true
    disp(['Marhale: ',num2str(epoch)])

    currentError = 0;
    for i=randperm(sizeX)
        net = x(i,1)*w(1) + x(i,2)*w(2) - theta;

        % Sigmoid Function
        output = 1 ./ (1 + exp(-net));
        
        w(1) = w(1) + eta*(x(i,3) - output)*output*(1-output)*x(i,1);
        w(2) = w(2) + eta*(x(i,3) - output)*output*(1-output)*x(i,2);
        theta = theta + eta*(x(i,3) - output)*output*(1-output)*(-1);

        % Error handling
        currentError = currentError + 1/2 * (x(i,3) - output)^2 ;

        % Print outputs
        T = table(x(i,1),x(i,2),theta,w(1),w(2),output,x(i,3),'VariableNames',{'X1','X2','THETA','W1','W2','OUTPUT','TARGET'}); 
        disp(T)
        
        % Plot boundry condition
        xP = -0.5 : 0.1 : 1.5 ;
        yP = (-w(1) ./ (w(2)))*xP + (theta ./ w(2)) ;
        h1 = plot(xP,yP,'b','LineWidth',1);
        pause(0.03)
        delete(h1)

    end

    % Error handling
    if abs(currentError - oldError) <= epsilon
        break
    end

    oldError = currentError;

    epoch = epoch + 1;
end

% Plot last boundry condition
plot(xP,yP,'k','LineWidth',2)