function [lambda,validation_lambda_error] = lasso_lambda_selector(X,y)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    lambda_set = [0;10;20;30;40;50;60;70;80;90;100];
    validation_lambda_error = zeros(11,1);
    
    for i = 1:11
        temp_lambda = lambda_set(i);
        %get one lambda from the lambda set
        vali_temp_error = 0;
        
        f = cvpartition(y, 'KFold', 10);
        
        for j = 1:10 %10 folds cross validation
            train_inds = f.training(j);
            test_inds = f.test(j);
            w = lasso(X(train_inds,:), y(train_inds),0,temp_lambda);
            vali_error = mean((y(test_inds) - X(test_inds,:)*w).^2);
            vali_temp_error = vali_temp_error + vali_error;
        end
        validation_lambda_error(i) = vali_temp_error./10;
    end
    [~,index] = min(validation_lambda_error);
    lambda = lambda_set(index);
end