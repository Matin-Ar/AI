close all;
clear;
clc;

% Input
numberOfCenters = input('Enter number of centers: '); % numberOfCenters = 3
spread = input('Enter spread: '); % spread = 1

% Load iris dataset
load fisheriris
[~, ~, target] = unique(species);
[sizexM, sizeyM] = size(meas);

% Split data to training and test set 
randArray = randperm(sizexM);
splitNumber = round(sizexM * 0.8);
trainInput = zeros(splitNumber,sizeyM);
trainOutput = zeros(splitNumber,1);
testInput = zeros(sizexM - splitNumber,sizeyM);
testOutput = zeros(sizexM - splitNumber,1);
for i= 1:splitNumber
    trainInput(i,:) = meas(randArray(i),:);
    trainOutput(i,:) = target(randArray(i),:);
end
for i= splitNumber+1:sizexM
    testInput(i-splitNumber,:) = meas(randArray(i),:);
    testOutput(i-splitNumber,:) = target(randArray(i),:);
end

% K-mean clustering
[~,center] = kmeans(trainInput,numberOfCenters);
[trainSize,~]=size(trainInput);

% Set phi
for i = 1:trainSize
    for j = 1:numberOfCenters
        phi(i,j) = (-(norm(trainInput(i,:) - center(j,:))) .^ 2) / (2* (spread .^ 2));
    end
end

% Set weights
w = (inv(phi.' * phi) * phi.') * trainOutput;

% Validate test
[testSize,~]=size(testInput);

for i = 1:testSize
    for j = 1:numberOfCenters
        phiTest(i,j) = (-(norm(testInput(i,:) - center(j,:))) .^ 2) / (2* (spread .^ 2));
    end
end

output = round(phiTest*w);

validateError = 0;
for i = 1:testSize
    if output(i) ~= testOutput(i)
        validateError = validateError + 1;
    end
end

validateAccuracy = (testSize - validateError)*100 / testSize;

% Print
disp('Centers:')
disp(center)
disp('Last neuron weights:')
disp(w)
fprintf('Validate test passed with %.2f%% accuracy.\n',validateAccuracy)