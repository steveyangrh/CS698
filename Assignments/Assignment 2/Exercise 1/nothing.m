
close all
clear
clc
load('MNIST_X_test.mat');
load('MNIST_X_train.mat');
load('MNIST_y_train.mat');
load('MNIST_y_test.mat');


%Xtest = Xtest(1:10000,:);
%ytest = ytest(1:10000,:);
tic
[newLabel2] = myknn3(Xtrain,ytrain,Xtest,0,3);
false_rate = nnz(newLabel2-ytest)./length(newLabel2);

toc



%{
tic
[newLabel3] = myknn5(Xtrain,ytrain,Xtest,0,1);
toc
%}







