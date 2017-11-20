function [lambda,validation_lambda_error] = winnow_lambda_selector(X,y,lambda_set)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    [a,b] = size(lambda_set);
    lsize = max(a,b);
    validation_lambda_error = zeros(lsize,1);
    
    for i = 1:lsize
        temp_lambda = lambda_set(i);
        %get one lambda from the lambda set
        vali_temp_error = 0;
        
        f = cvpartition(y, 'KFold', 10);
        
        for j = 1:10 %10 folds cross validation
            train_inds = f.training(j);
            test_inds = f.test(j);
            %w = ridge(X(train_inds,:), y(train_inds),0,temp_lambda);
            [w,~] = winnow2(X(train_inds,:),y(train_inds),0,0,temp_lambda,500);
            prediction = X(test_inds,:)*w;
            prediction(prediction<=0) = -1;
            prediction(prediction>0) = 1;
            temp_mistake = length(y(test_inds))-nnz(y(test_inds) - prediction);
            
            % calculate the validation error
            vali_temp_error = vali_temp_error + temp_mistake;
            % sum up the validation error
        end
        validation_lambda_error(i) = vali_temp_error./10;
        % divide the validation error by 10
    end
    [~,index] = min(validation_lambda_error);
    lambda = lambda_set(index);
    % return lambda that produces the smallest error
end