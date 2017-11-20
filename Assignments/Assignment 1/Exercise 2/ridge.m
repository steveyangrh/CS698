function [w] = ridge(X,y,w,lambda)
%This function uses idge regression to calculate w
%w here is [w,b]
    
    K = X' * X;
    [numR,~] = size(K);
    % numR stores the number of data points
    K = K + lambda.* eye(numR);
    % add an identity matrix to it
    Q = X'*y;
    w = K\Q;
    % get w by solving the linear system

end

