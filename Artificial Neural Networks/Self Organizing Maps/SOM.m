close all;
clear;
clc;

% Fprint
fprintf('Hello, This Kohonen Network Designed by "Matin Araghi". \n')
fprintf('Our network size is (N*M) and learning rates are: (a_min & a_max). \n')
fprintf('We suggest you these parameters: \n')
fprintf('n->7 , m->7 , a_min->0.1 , a_max->1.5 \n')
fprintf('Anyway, our network is completely tunable. \n \n')
fprintf('Please enter parameters: \n')

% Input
n=input('Enter n:'); % n=7;
m=input('Enter m:'); % m=7;
a_min=input('Enter a_min:'); % a_min=0.1;
a_max=input('Enter a_max:'); % a_max=1.5;

% Load iris dataset
load fisheriris
data = meas(:,1:2)/10;
% data = [-0.6 0.1
%         -0.4 0.3
%         -0.3 -0.3
%         -0.2 0.2
%         0.1 0.5
%         0.2 -0.2
%         0.2 -0.5
%         0.3 0.3
%         0.4 0.6];
[dataSize,~]=size(data);

% Plot
f1 = figure();
f1.WindowState = "maximized";
hold on
scatter(data(:,1),data(:,2),"blue","filled")

% Set wights
w=rand(n*m,2);
t_max=dataSize;

% Training
% for epoch=1:100
for t=1:dataSize
    % Find winner
    for i=1:n*m
        arg(i)=norm(data(t,:)-w(i,:));
    end
    [~,c]=min(arg);
    % Find winner's position
    c_index_i = ceil(c/m);
    c_index_j = mod(c,m);
    if c_index_j == 0
        c_index_j = m;
    end
    c_index = [c_index_i,c_index_j];
    % Training
    eta=(a_max - a_min)*((t_max - t)/(t_max - 1))+a_min;
    for i=1:n
        for j=1:m
            d = dist([c_index;i,j],t);
            w((i-1)*m+j,:) = w((i-1)*m+j,:) + eta*d*(data(t,:) - w((i-1)*m+j,:));
        end
    end
    matinplot(w,n,m,t,dataSize)
end
% end


% Functions
% dist function
function d = dist(c_j,t)
    d_jc=pdist(c_j,"cityblock");
    d=exp(-(d_jc.^2)/(2*(sigma(t).^2)));
end

% sigma function
function s=sigma(t)
    s = 2*exp(-t/150);
end

% matinplot function
function matinplot(w,n,m,t,dataSize)
    s1 = scatter(w(:,1),w(:,2),"red","filled");
    check =1;
    for i=1:n
        for j=1:m
            for k=i:n
                for z=1:m
                    d = pdist([i,j;k,z]);
                    if d==1
                        p1(check) = plot([w((i-1)*m+j,1);w((k-1)*m+z,1)],[w((i-1)*m+j,2);w((k-1)*m+z,2)],'black');
                        check = check + 1;
                    end
                end
            end
        end
    end
    drawnow
    if t~=dataSize
        delete(s1)
        delete(p1)
    end
end