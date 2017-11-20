clear
clc
warning('off','all')

%% Exercise 2 question 1
%{
load('housing_X_train.mat');
load('housing_y_train.mat');
[numR,~] = size(Xtrain);
Xtrain = [Xtrain ones(numR,1)];
% appending ones to X
[lambda,validation_lambda_error] = ridge_lambda_selector(Xtrain,ytrain);
% calculate the validation error
load('housing_X_test.mat');
load('housing_y_test.mat');
[numR,~] = size(Xtest);
Xtest = [Xtest ones(numR,1)];
train_error = zeros(11,1);
test_error = zeros(11,1);
lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
disp('Exercise 2, question 1:');
for i = 1:11
    w = ridge(Xtrain,ytrain,0,lambda_set(i));
    prediction_test = Xtest * w;
    error_test =  mean((prediction_test - ytest).^2);
    test_error(i) = error_test;
    prediction_train = Xtrain * w;
    error_train =  mean((prediction_train - ytrain).^2);
    train_error(i) = error_train;
    X = sprintf('lambda %d, validation error%10.5f, training error%10.5f, test error %10.5f, density of w %d',lambda_set(i),validation_lambda_error(i),train_error(i),test_error(i),nnz(w)/prod(size(w)));
    disp(X);
end
%}
% calculate the test error
% calculate the training error


%% Exercise 2 question 2
%{
clear
load('housing_X_train.mat');
load('housing_y_train.mat');
[numR,~] = size(Xtrain);
Xtrain = [Xtrain ones(numR,1)];
indx = randi(numR,1);
%ytrain(indx) = ytrain(indx).*10^(3);
Xtrain(indx,1:end-1) = Xtrain(indx,1:end-1).*10^(3);
[lambda,validation_lambda_error] = ridge_lambda_selector(Xtrain,ytrain);
% calculate the validation error
load('housing_X_test.mat');
load('housing_y_test.mat');
[numR,~] = size(Xtest);
Xtest = [Xtest ones(numR,1)];
train_error = zeros(11,1);
test_error = zeros(11,1);
lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
disp('Exercise 2, question 2:');
for i = 1:11
    w = ridge(Xtrain,ytrain,0,lambda_set(i));
    prediction_test = Xtest * w;
    error_test =  mean((prediction_test - ytest).^2);
    test_error(i) = error_test;
    prediction_train = Xtrain * w;
    error_train =  mean((prediction_train - ytrain).^2);
    train_error(i) = error_train;
    X = sprintf('lambda %d, validation error%15.5f, training error%15.5f, test error %15.5f, density of w %d',lambda_set(i),validation_lambda_error(i),train_error(i),test_error(i),nnz(w)/prod(size(w)));
    disp(X);
end
%}

%% Exercise 2 question 3
%{
close all
clc
clear
load('housing_X_train.mat');
load('housing_y_train.mat');
load('housing_X_test.mat');
load('housing_y_test.mat');
[numR,~] = size(Xtrain);
Xtrain_append = randn(numR,1000);
Xtrain = [Xtrain Xtrain_append ones(numR,1)];

[numR,~] = size(Xtest);
Xtest_append = randn(numR,1000);
Xtest = [Xtest Xtest_append ones(numR,1)];
train_error = zeros(11,1);
test_error = zeros(11,1);
lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
[lambda,validation_lambda_error] = ridge_lambda_selector(Xtrain,ytrain);
disp('Exercise 2, question 3:');
for i = 1:11
    w = ridge(Xtrain,ytrain,0,lambda_set(i));
    prediction_test = Xtest * w;
    error_test =  mean((prediction_test - ytest).^2);
    test_error(i) = error_test;
    prediction_train = Xtrain * w;
    error_train =  mean((prediction_train - ytrain).^2);
    train_error(i) = error_train;
    X = sprintf('lambda %d, validation error%15.5f, training error%15.5f, test error %15.5f, density of w %d',lambda_set(i),validation_lambda_error(i),train_error(i),test_error(i),nnz(w(end-999:end))/(prod(size(w))-14));
    disp(X);
end
%}

%% Exercise 2 question 4
%{
close all
clc
clear
load('housing_X_train.mat');
load('housing_y_train.mat');
load('housing_X_test.mat');
load('housing_y_test.mat');
[numR,~] = size(Xtrain);
Xtrain = [Xtrain ones(numR,1)];
[numR,~] = size(Xtest);
Xtest = [Xtest ones(numR,1)];
[lambda,validation_lambda_error] = lasso_lambda_selector(Xtrain,ytrain);
lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
disp('Exercise 2, question 4:');
for i = 1:11
    w = lasso(Xtrain,ytrain,0,lambda_set(i));
    prediction_test = Xtest * w;
    error_test =  mean((prediction_test - ytest).^2);
    test_error(i) = error_test;
    prediction_train = Xtrain * w;
    error_train =  mean((prediction_train - ytrain).^2);
    train_error(i) = error_train;
    X = sprintf('lambda %d, validation error%15.5f, training error%15.5f, test error %15.5f, density of w %d',lambda_set(i),validation_lambda_error(i),train_error(i),test_error(i),nnz(w)/prod(size(w)));
    disp(X);
end
%}

%% Exercise 2 question 5

close all
clc
clear
load('housing_X_train.mat');
load('housing_y_train.mat');
load('housing_X_test.mat');
load('housing_y_test.mat');
[numR,~] = size(Xtrain);
Xtrain_append = randn(numR,1000);
Xtrain = [Xtrain Xtrain_append ones(numR,1)];
[numR,~] = size(Xtest);
Xtest_append = randn(numR,1000);
Xtest = [Xtest Xtest_append ones(numR,1)];
[lambda,validation_lambda_error] = lasso_lambda_selector(Xtrain,ytrain);
lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
disp('Exercise 2, question 3:');
for i = 1:11
    w = lasso(Xtrain,ytrain,0,lambda_set(i));
    prediction_test = Xtest * w;
    error_test =  mean((prediction_test - ytest).^2);
    test_error(i) = error_test;
    prediction_train = Xtrain * w;
    error_train =  mean((prediction_train - ytrain).^2);
    train_error(i) = error_train;
    X = sprintf('lambda %d, validation error%15.5f, training error%15.5f, test error %15.5f, density of w %d',lambda_set(i),validation_lambda_error(i),train_error(i),test_error(i),nnz(w(end-999:end))/(prod(size(w))-14));
    disp(X)
end

