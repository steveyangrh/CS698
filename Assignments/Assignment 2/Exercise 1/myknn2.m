function [newLabels] = myknn2(X,y,Z,d,k)
% myknn2 is a function that works well with large data set
% myknn2 works much faster than myknn
% the default distance function is euclidean distance
    
    [numR,~] = size(Z);
    %numR is the number of data points
    %numR is m
    [numP,~] = size(X);
    %numP is n

    newLabels = [];
    for i = 1:numR
        z = Z(i,:);
        %Z_new = repelem(z,numP,1);
        switch d
            case {'Chebyshev','chebyshev'}
                dis = max(abs(X-z),[],2);
            case {'Manhattan','manhattan'}
                dis = sum(abs(X-z),2);
            otherwise
                %dis = sqrt(sum((X-z).^2,2));
                dis  = X'*X - 2*(X'*z) +z'*z;
        end
        %dis stores the all the distances
        [~,I] = sort(dis,'ascend');
        len=min(k,length(dis));
        tempLable=mode(y(I(1:len)));
        newLabels = [newLabels;tempLable];
    end
    
    
end