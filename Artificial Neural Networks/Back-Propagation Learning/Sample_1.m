clear 
close all
clc

% Input
x=[0 0 1 1
   0 1 0 1];
% Target
t=[0 1 1 0];

[~,sizeX] = size(x);

% Config Values
x1=[x;-1 -1 -1 -1];
w1 = rand(2,3);
w2 = rand(1,3);
eta = 0.7;
maxEpoch = 2500;
epsilon = 0.00001;
oldError = 0;

% Plot
hold on

% Training
for i=1:maxEpoch
    disp(['Iteration: ',num2str(i)])

    T = table(zeros(sizeX,1),zeros(sizeX,1),zeros(sizeX,3),zeros(sizeX,3),zeros(sizeX,3),zeros(sizeX,1),zeros(sizeX,1),'VariableNames', {'X1','X2','W1_1','W1_2','W2_1','OUTPUT','TARGET'});

    dw1=0;
    dw2=0;
    currentError = 0;
    for j=randperm(sizeX)
        net1 = w1*x1(:,j);
        output = 1 ./ (1 + exp(-net1));
        output1 = [output;-1];
        net2 = w2*output1;
        output2 = 1 ./ (1 + exp(-net2));
        
        delta2_1 = (t(j)-output2)*output2*(1-output2);
        delta1_1 = (delta2_1*w2(1))*output(1)*(1-output(1));
        delta1_2 = (delta2_1*w2(2))*output(2)*(1-output(2));

        dw2_1_new = eta*delta2_1*output1;
        dw2 = dw2 + dw2_1_new;
        w2 = w2 + dw2';

        dw1_1_new = eta*delta1_1*(x1(:,j));
        dw1_2_new = eta*delta1_2*(x1(:,j));
        dw1 = dw1 + [dw1_1_new,dw1_2_new];
        w1 = w1 + dw1';
        
        % Error handling
        currentError = currentError + (1/2 * ((t(j) - output2)^2));
        
        % Print inputs and corresponding outputs
        T.X1(j,1)=x1(1,j);
        T.X2(j,1)=x1(2,j);
        T.W1_1(j,:)=w1(1,:);
        T.W1_2(j,:)=w1(2,:);
        T.W2_1(j,:)=w2;
        T.OUTPUT(j,1)=output2;
        T.TARGET(j,1)=t(j);
    end

    disp(T)
    matinplot(currentError,i)

    if abs(currentError - oldError) <= epsilon
        break
    end
    
    oldError = currentError;
    clc
end

% Show a 3D plot of network output vs. inputs
figure
hold on
view(3)
p3=0:0.03:1;
for ii=p3
    for jj=p3
        kk = 1 ./ (1 + exp(-w2*[1 ./ (1 + exp(-w1*[ii;jj;-1]));-1]));
        plot3(ii,jj,kk,'.')
    end
end

% Plot the error value vs. epoch during the training session
function matinplot(currentError,i)
xlim([0 i])
ylim([0 1])
xlabel('Epoch');
ylabel('Error')
plot(i,currentError,'.')
drawnow
end