clear
close
clc


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% loading data
load('data_batch_1.mat', 'data');
trainX = data;
trainX = double(trainX);
load('data_batch_1.mat', 'labels');
trainy = labels;
trainy = double(trainy);
load('test_batch.mat', 'data');
testX = data;
testX = double(testX);
load('test_batch.mat', 'labels');
testy = labels;
testy = double(testy);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set the default values
lambda = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% append 1 to both trainX and testX
[rowX,colX] = size(trainX);
trainX = trainX./max(max(trainX));
trainX = [trainX ones(rowX,1)];
[rowX,colX] = size(testX);
testX = testX./max(max(testX));
testX = [testX ones(rowX,1)];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


errors = [];
for lambda = 1:5

    Y = [];

    %% Training process
    for i = 0:9
        for j = 0:9
            if i ~= j
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% modify y
                indI = find(trainy==i);
                indJ = find(trainy==j);
                tempXI = trainX(indI,:);
                tempXJ = trainX(indJ,:);
                tempyI = ones(length(indI),1);            
                tempyJ = -ones(length(indJ),1);
                tempX = [tempXI;tempXJ];
                tempy = [tempyI;tempyJ];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% trianing process
                [w] = binaryLR(tempX,tempy,lambda);
                W(:,:,i+1,j+1) = w;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %% trianing process
                y_new = (w'*testX')';
                y_new(y_new>0) = i;
                y_new(y_new<0) = j;
                Y = [Y y_new];
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            end
        end
    end

    ypredict = mode(Y,2);
    error_rate = nnz(testy - ypredict)./length(ypredict);
    errors = [errors error_rate];
end

plot(1:5, errors);
title('error rates vs lambda');
xlabel('lambda');
ylabel('error rate');
