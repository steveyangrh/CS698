function [newLabels] = myknn4(X,y,Z,d,k)
% myknn3 is a function that works well with large data set
% myknn3 works much faster than myknn
% the default distance function is euclidean distance
% difference between myknn2 and myknn3: myknn3 uses parfor
    [m,~] = size(Z);
    %numP is n

    newLabels = [];
    numL = 1;
    numZ  = m./numL;
    %numL is the number of loops
    %numZ is the number of points calculated in each loop
    y = y';
    numT = 60;
    numTrain  = 60000./numT;

    for i = 1:numL
        z = Z(((i-1)*numZ+1):i*numZ,:);
        [numR,~] = size(z);
        dis = [];
        disp('hello');
        for j = 1:numT
            X_temp = X(((j-1)*numTrain+1):j*numTrain,:);
            X_new = repmat(X_temp,[numR,1]);
            [numP,~] = size(X_temp);
            Z_new = repelem(z,numP,1);
            switch d
                case {'Chebyshev','chebyshev'}
                    temp_dis = max(abs(X_new-Z_new),[],2);
                case {'Manhattan','manhattan'}
                    temp_dis = sum(abs(X_new-Z_new),2);
                otherwise
                    temp_dis = sqrt(sum((X_new-Z_new).^2,2));
            end
            dis = [dis;temp_dis];
        end
        %dis stores the all the distances
        dis_new = reshape(dis,[60000,numR]);
        [~,I] = sort(dis_new,'ascend');
        tempLable = mode(y(I(1:k,:)));
        if k ==1 
            tempLable = mode(y(I(1:k,:))',1);
        end       
        newLabels = [newLabels;tempLable'];
    end
    
    
end