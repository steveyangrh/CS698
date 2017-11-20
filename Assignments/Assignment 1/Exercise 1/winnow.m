% September 13, 2017
% Assignment 1 for CS698, UWaterloo
% Ronghao Yang

function [W,b,mistake] = winnow(X,y,W,b,step_size,max_pass)
    %transpose X and Y so that each column represent a data point
    %rows represent the features
    
    [numR,numC] = size(X);
    %numR(n) is the number of rows of X, which is the number of data points
    %numC(d) is the number of columns of X, which is the number of features
    
    W = ones(numC,1)./(numC+1);
    %initialize W to be 1

    b = 1/(numC+1);
    %initialize b
    
    for t = 1:max_pass
        mistake(t) = 0;
        for i = 1:numR
            a = X(i,:);
            % a is a 1 by 57 vector
            % W is a 57 by 1 vector
            if (a*W+b).*y(i) <= 0
                W = W.* exp(step_size.*a'.*y(i));
                b = b.* exp(step_size.*y(i));
                s = b + sum(W);
                W = W./s;
                b = b./s;
                %updating W is a mistake is made
                mistake(t) = mistake(t) + 1;
            end
        end
    end
end