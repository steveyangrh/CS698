clear
close
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% loading data
load('train_X_dog_cat.mat');
trainX = M;
trainX = double(trainX);
load('train_y_dog_cat.mat');
trainy = M;
trainy = double(trainy);
load('test_X_dog_cat.mat');
testX = M;
testX = double(testX);
load('test_y_dog_cat.mat');
testy = M;
testy = double(testy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% append 1 to both trainX and testX
[rowX,colX] = size(trainX);
trainX = trainX./max(max(trainX));
trainX = [trainX ones(rowX,1)];
[rowX,colX] = size(testX);
testX = testX./max(max(testX));
testX = [testX ones(rowX,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the default values
lambda = 5;
kernel = 'polynomial';
epsilon = 400;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

errors = [];
for lambda = 1:1:5
    [alpha] = KbinaryLR(trainX,trainy,lambda,kernel,epsilon);
    [ypredict,~] = kpredict(alpha,trainX,testX,kernel,epsilon);
    error_rate = nnz(testy - ypredict)./length(ypredict);
    errors = [errors error_rate];
end
plot(1:5, errors);
title('Gaussian Kernel: error rates vs epsilon, lambda=3');
xlabel('epsilon');
ylabel('error rate');