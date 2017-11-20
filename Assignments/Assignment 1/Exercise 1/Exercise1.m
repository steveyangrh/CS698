% September 13, 2017
% Assignment 1 for CS698, UWaterloo
% Ronghao Yang


%% Exercise 1 question 1
%{
clear
close all
clc


load('spambase_X.mat');
load('spambase_Y.mat');
max_pass = 500;
% loading data matrices from the mat files and initialize the iterations
[W,b,mistake] = perceptron(X,y,0,0,max_pass);
% W and b are initialized to 0

subplot(2,3,1);
plot(1:max_pass,mistake,'o');
title('Exercise 1 question 1');
xlabel('# of passes');
ylabel('# of mistakes');
%}

%% Exercise 1 question 2
%{
load('spambase_X.mat');
load('spambase_Y.mat');
max_pass = 500;
% loading data matrices from the mat files and initialize the iterations
[W,b,mistake] = perceptron2(X,y,0,0,max_pass);
% W and b are initialized to 0

subplot(2,3,2);
plot(1:max_pass,mistake,'o');
title('Exercise 1 question 2');
xlabel('# of passes');
ylabel('# of mistakes');
%}


%% Exercise 1 question 4

clear
clc
load('spambase_X.mat');
load('spambase_Y.mat');
max_pass = 500;
% loading data matrices from the mat files and initialize the iterations

step_size = 0.2;
[W,b,mistake] = winnow(X,y,0,0,step_size,max_pass);
% W and b are initialized to 0
subplot(2,3,3);
plot(1:max_pass,mistake,'o');
title('Exercise 1 question 4');
xlabel('# of passes');
ylabel('# of mistakes');


%% Exercise 1 question 5
%{
clear
clc
load('spambase_X.mat');
load('spambase_Y.mat');
[numR,numC] = size(X);
X_1 = ones(numR,1);
X_1 = [X X_1];
X_new = [X_1 -X_1];
max_pass = 500;
% loading data matrices from the mat files and initialize the iterations
max_x = max(X_1(:));
lambda_set = 1/max_x: 1/max_x: 10/max_x;
[lambda,validation_lambda_error] = winnow_lambda_selector(X_new,y,lambda_set);
[W,mistake] = winnow2(X_new,y,0,0,lambda,max_pass);
% W and b are initialized to 0

subplot(2,3,4);
plot(1:max_pass,mistake,'o');
title('Exercise 1 question 5');
xlabel('# of passes');
ylabel('# of mistakes');
%}



%% Exercise 1 question 6
%{
clear
clc
load('spambase_X.mat');
load('spambase_Y.mat');
[numR,numC] = size(X);
max_pass = 500;
%X = X./max(X(:));
%normalization

new_F = randi([-1 1],numR,100);
X_per = [X new_F];
X_1 = ones(numR,1);
X_win = [X_per X_1];
%X_per is X with 100 new features
%X_per is used for perceptron algorithm
%X_win is used for winnow algorithm
[~,~,mistake1] = perceptron(X_per,y,0,0,max_pass);
X_win = [X_win -X_win];
lambda = 2/max(X(:));
[~,mistake2] = winnow2(X_win,y,0,0,lambda,max_pass);

subplot(2,3,5);
a1 = plot(1:max_pass,mistake1,'color','r');
title('Exercise 1 question 6');
xlabel('# of passes');
ylabel('# of mistakes');
hold on;
a2 = plot(1:max_pass,mistake2,'color','b');
title('Exercise 1 question 6');
xlabel('# of passes');
ylabel('# of mistakes');
hold on;
M1 = 'Perceptron';
M2 = 'Winnow';
legend([a1;a2], M1, M2);
%}

