close all
clear
clc

K = 1;

n = 10000;
Xtrain = mvnrnd([1 2 3], diag([4 5 6]), n);
[c,u,S] = GMM(Xtrain,K);