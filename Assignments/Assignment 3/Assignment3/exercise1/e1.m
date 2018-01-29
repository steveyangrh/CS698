clear
close all
clc

%{
load('data_batch_5.mat', 'data');
trainX = data;
trainX = double(trainX);
load('data_batch_5.mat', 'labels');
trainy = labels;
trainy = double(trainy);
load('test_batch.mat', 'data');
testX = data;
testX = double(testX);
load('test_batch.mat', 'labels');
testy = labels;
testy = double(testy);
%}


load('train_X_dog_cat.mat');
trainX = M;
load('train_y_dog_cat.mat');
trainy = M;
load('test_X_dog_cat.mat');
testX = M;
load('test_y_dog_cat.mat');
testy = M;


trainX = trainX./max(max(trainX));
[w] = binaryLR(trainX,trainy,0.01);
y_new = w'*testX';
y_new(y_new<0) = -1;
y_new(y_new>0) = 1;
error_rate = nnz(testy - y_new')./length(y_new)