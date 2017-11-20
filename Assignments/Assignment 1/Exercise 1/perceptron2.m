function [W,b,mistake] = perceptron2(X,y,W,b,max_pass)

    %transpose X and Y so that each column represent a data point
    %rows represent the features
    
    [numR,numC] = size(X);
    %numR is the number of rows of X, which is the number of data points
    %numC is the number of columns of X, which is the number of features 

    W = zeros(numC,1);
    %initialize W to be 0
 
    for t = 1:max_pass
        mistake(t) = 0;
        for i = 1:numR
            a = X(i,:); 
            
            W = W + a'.*y(i);
            b = b + y(i);
            if (a*W+b)*y(i) <= 0         
                %{
                W = W + a'.*y(i);
                b = b + y(i);
                %}
                %updating W is a mistake is made
                mistake(t) = mistake(t) + 1;
            end
            
        end
    end
    
end