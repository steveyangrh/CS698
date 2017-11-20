% September 13, 2017
% Assignment 1 for CS698, UWaterloo
% Ronghao Yang

function [W,b,mistake] = perceptron(X,y,W,b,max_pass)
% this function is a implementation of perceptron algorithm
    
    X = X';
    Y = y';
    %transpose X and Y so that each column represent a data point
    %rows represent the features
    
    [numR,numC] = size(X);
    %numR is the number of rows of X
    %numC is the number of columns of X
    
    for i = 1:numC
        X(:,i) = X(:,i) * Y(i);
    end
    A = [X;Y];
    %form the matrix A, A = [a1,a2,a3..an], ai = [YiXi;Yi]  
     
    [numR,numC] = size(A);
    %numR is the number of rows of A
    %numC is the number of columns of A
    
    W = zeros(1,numR);
    %initialize W to be 0
 
    for t = 1:max_pass
        mistake(t) = 0;
        for i = 1:numC
            a = A(:,i);
            if W*a <= 0
                W = W + a';
                %updating W is a mistake is made
                mistake(t) = mistake(t) + 1;
            end
        end
    end
    
    b = W(end);
    W = W(1:end-1);
end