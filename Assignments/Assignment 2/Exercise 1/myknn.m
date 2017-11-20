function [newLabels] = myknn(X,y,Z,d,k)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    [numR,~] = size(Z);
    %numR is the number of data points
    %numC is the number of features
    [numP,~] = size(X);
    
    X_new = repmat(X,[numR,1]);
    Z_new = repelem(Z,numP,1);

    switch d
        otherwise
            dis = sqrt(sum((X_new-Z_new).^2,2));
    end
    %dis stores the all the distances
    
    newLabels = [];
    for i = 1:numR
        startI = (i-1)*numP+1;
        endI = i*numP;
        oneDists = dis(startI:endI);
        [~,I] = sort(oneDists,'ascend');
        len=min(k,length(oneDists));
        tempLable=mode(y(I(1:len)));
        newLabels = [newLabels;tempLable];
    end
    
end

