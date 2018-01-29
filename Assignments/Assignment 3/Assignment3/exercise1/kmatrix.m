function [K] = kmatrix(X1,X2,kernel,epsilon)
    
    switch nargin
        case 3
            sigma= 400;
    end

    switch kernel
        case 'linear'
            K = X1*X2';
        case 'polynomial'
            K = (1+X1*X2').^5;
        case 'gaussian'
            [n, ~] = size(X1);
            [m, ~] = size(X2);
            K = exp((repmat(diag(X2 * X2'),1,n)-2*X2*X1'+repmat(diag(X1 * X1')',m,1))/-epsilon);
            K = K';
    end
end

