close all
clear
clc
warning('off','all')



load('MNIST_X_train.mat');
load('MNIST_y_train.mat');
load('MNIST_X_test.mat');
load('MNIST_y_test.mat');
%loading the dataset

figure
K = 5;
error_rates_PCA = [];
for pca_value = 10:10:100
    [error_rate, ~] = prediction(Xtrain,ytrain,Xtest,ytest,K,pca_value);
    error_rates_PCA = [error_rates_PCA error_rate];
end
plot(10:10:100,error_rates_PCA);
title('Error Rate vs pca values, K=5');
xlabel('pca values');
ylabel('Error Rate');

figure
pca_value = 40;
%set the pre-values
error_rates_K = [];
for K = 1:2:9
    [error_rate, ~] = prediction(Xtrain,ytrain,Xtest,ytest,K,pca_value);
    error_rates_K = [error_rates_K error_rate];
end
plot(1:2:9,error_rates_K);
title('Error Rate vs K values, pca value=40');
xlabel('K values');
ylabel('Error Rate');


display('When K = 5, pca_value = 100, the error rate is:');
[error_rate, ~] = prediction(Xtrain,ytrain,Xtest,ytest,5,100)



