function [prediction,K] = kpredict(alpha,trainX,testX,kernel,epsilon)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here
    switch kernel
        case 'linear'
            K = trainX*testX';
        case 'polynomial'
            K = (1+trainX*testX').^5;
        case 'gaussian'
            %{
            D = sqdist(trainX',testX').^2;
            K = exp(-D./epsilon);
            %}
            K = kmatrix(trainX,testX,'gaussian',epsilon);
    end
    %K has size 1953 * 2000
    [a,b] = size(K)
    e = exp(-(alpha'*K)');
    p = 1./(1+e);
    p(p<=0.5) = -1;
    p(p>0.5) = 1;
    prediction = p;
end

