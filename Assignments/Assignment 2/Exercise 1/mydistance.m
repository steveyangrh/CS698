function [D] = mydistance(X,z)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    [m,~] = size(z);
    [n,~] = size(X);

    A = sum((X.^2),2);
    A = repmat(A,1,m);
    B = X*z';
    C = sum((z.^2),2);
    C = repmat(C,1,n);
    C = C';
    %D = A - 2*B - C;
    D = A - 2*B - C;
end

