function [newLabels] = myknn3(X,y,Z,d,k)
% myknn3 is a function that works well with large data set
% myknn3 works much faster than myknn
% the default distance function is euclidean distance
% difference between myknn2 and myknn3: myknn3 uses parfor
    [m,~] = size(Z);
    [numP,~] = size(X);
    %numP is n

    newLabels = [];
    numL = 10;
    numZ  = m./numL;
    %numL is the number of loops
    %numZ is the number of points calculated in each loop
    for i = 1:numL
        z = Z(((i-1)*numZ+1):i*numZ,:);
        [numR,~] = size(Z);
        switch d
            case {'Chebyshev','chebyshev'}
                dis = max(abs(X_new-Z_new),[],2);
            case {'Manhattan','manhattan'}
                dis = sum(abs(X_new-Z_new),2);
            otherwise
                dis = mydistance(X,z);
        end
        %dis stores the all the distances
        
        %dis_new = reshape(dis,[numP,numR]);
        [~,I] = sort(dis,'ascend');
        tempLable = mode(y(I(1:k,:)));
        if k ==1 
            tempLable = mode(y(I(1:k,:))',1);
        end
        newLabels = [newLabels;tempLable'];
        
    end
    
    
end