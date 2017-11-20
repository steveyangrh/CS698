function [w] = lasso(X,y,w,lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    error = inf;
    tolerrence = 0.001;
    [numR,numC] = size(X);
    w = zeros(numC,1);
    while error > tolerrence
        w_old = w;
        for j =1:numC
            a = X(:,j);
            E = zeros(numR,1);
            for k = 1:numC
                if k~=j
                    E = E + X(:,k).*w(k);
                end
            end
            E = y-E;
            A = 0;
            B = 0;
            for k = 1:numR
                A = A + a(k).^2;
                B = B + a(k).*E(k);
            end
            if B>= 0
                signB = 1;
            end
            if B<0
                signB = -1;
            end
            w(j) = signB.*max(0,(abs(B)-lambda)./A);
        end
        error = mean((w_old - w).^2);
    end   
        
end

