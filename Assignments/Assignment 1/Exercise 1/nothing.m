% September 13, 2017
% Assignment 1 for CS698, UWaterloo
% Ronghao Yang

clear
close all
clc
inputX = load('./spambase/spambase.data');
X = inputX(:,1:end-1);
y = inputX(:,end);
y(y==0) = -1;

[W,b,mistake] = perceptron2(X,y,0,0,500);

