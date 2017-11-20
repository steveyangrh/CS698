function [k,validation_k_error] = knn_euclidean_selector(X,y,nfold,k_set)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    %lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
    k_len = length(k_set);
    validation_k_error = zeros(k_len,1);
    
    for i = 1:k_len
        temp_k = k_set(i);
        %get one lambda from the lambda set
        vali_temp_mistake = 0;
        
        f = cvpartition(y, 'KFold', nfold);
        % we use n fold cross validation
        
        for j = 1:nfold %n folds cross validation
            train_inds = f.training(j);
            test_inds = f.test(j);
            [newLabels] = myknn3(X(train_inds,:),y(train_inds),X(test_inds,:),0,temp_k);
            vali_temp_mistake = vali_temp_mistake + nnz(newLabels-y(test_inds))./length(newLabels);
        end
        validation_k_error(i) = vali_temp_mistake./nfold;
    end
    [~,index] = min(validation_k_error);
    k = k_set(index);
end