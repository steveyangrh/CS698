function [newLabels] = myknn5(X,y,Z,d,k)
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
        [numR,~] = size(z);
        disdata = repelem(z,numP,1)-repmat(X,[numR,1]);
        
        switch d
            case {'Chebyshev','chebyshev'}
                dis = max(abs(disdata),[],2);
            case {'Manhattan','manhattan'}
                dis = sum(abs(disdata),2);
            otherwise
                dis = sqrt(sum((disdata).^2,2));
        end
        %dis stores the all the distances
        
        dis_new = reshape(dis,[numP,numR]);
        [~,I] = sort(dis_new,'ascend');
        %[a,b] = size(y(I(1:k,:)))
        tempLable = mode(y(I(1:k,:)));
        if k ==1 
            tempLable = mode(y(I(1:k,:))',1);
        end
        newLabels = [newLabels;tempLable'];
        
    end
    
    
end