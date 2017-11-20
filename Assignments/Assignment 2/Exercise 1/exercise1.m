%% CS 698 Machine Learning
% Assignment 2
% Ronghao Yang

close all
clear
clc
load('MNIST_X_test.mat');
load('MNIST_X_train.mat');
load('MNIST_y_train.mat');
load('MNIST_y_test.mat');

nfold = 10;
k_set = [1;2;3;4;5];


%% euclidean
tic
[k,validation_k_error] = knn_euclidean_selector(Xtrain,ytrain,nfold,k_set);
[newLabels] = myknn3(Xtrain,ytrain,Xtest,0,k);
false_rate = nnz(newLabels-ytest)./length(newLabels);
toc
save('10X_[1,2,3,4,5]_euclidean');


%{
clear
load('MNIST_X_test.mat');
load('MNIST_X_train.mat');
load('MNIST_y_train.mat');
load('MNIST_y_test.mat');
nfold = 5;
k_set = [1;5;10;20;50];




%% Manhattan
tic
[k,validation_k_error] = knn_selector(Xtrain,ytrain,nfold,k_set,'manhattan');
[newLabels] = myknn2(Xtrain,ytrain,Xtest,'manhattan',k);
false_rate = nnz(newLabels-ytest)./length(newLabels);
toc

save('5X_[1,5,10,20,50]_manhattan');
clear
load('MNIST_X_test.mat');
load('MNIST_X_train.mat');
load('MNIST_y_train.mat');
load('MNIST_y_test.mat');
nfold = 5;
k_set = [1;5;10;20;50];

%% Chebyshev
tic
[k,validation_k_error] = knn_selector(Xtrain,ytrain,nfold,k_set,'chebyshev');
[newLabels] = myknn2(Xtrain,ytrain,Xtest,'chebyshev',k);
false_rate = nnz(newLabels-ytest)./length(newLabels);
toc

save('5X_[1,5,10,20,50]_chebyshev');
clear
clc
%}
