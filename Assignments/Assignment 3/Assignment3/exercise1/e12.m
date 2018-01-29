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

for lambda = 1:1:5

    W = [];
    procs = [];

    %% Training process
    for i = 0:9
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% modify y
        tempy = trainy;
        tempy(tempy ~= i) = -1;
        tempy(tempy == i) = 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% trianing process
        [w] = binaryLR(trainX,tempy,lambda);
        W = [W w];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% trianing process
        proc = 1./(1+exp(-(w'*testX')'));
        procs = [procs proc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end


    [~,ypredict] = max(procs,[],2);
    ypredict = ypredict - 1;
    error_rate = nnz(testy - ypredict)./length(ypredict);
    errors = [errors error_rate];

end

plot(1:5, errors);
title('error rates vs lambda');
xlabel('lambda');
ylabel('error rate');