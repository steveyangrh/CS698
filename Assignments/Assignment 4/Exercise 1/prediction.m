function [error_rate,y_prediction] = prediction(Xtrain,ytrain,Xtest,ytest,K,pca_value)
    [n,d] = size(Xtrain);
    %get the size of the training matrix

    C = cell(10);
    U = cell(10);
    S = cell(10);
    P = [];
    PCA = cell(10);
    %initialize the cells and arrays




    for i = 0:9
        tempX = Xtrain(ytrain==i,:);
        [pc,s,l,t,e] = pca(tempX);
        PCA{i+1} = pc(:,1:pca_value);
        tempX = tempX*PCA{i+1};
        [c,u,s] = GMM(tempX,K);
        C{i+1} = c;
        U{i+1} = u;
        S{i+1} = s;
        P(i+1) = sum(ytrain == i)./n;
    end

    resP = [];
    %resP contains the results

    for i = 0:9
        pa = PCA{i+1};
        tempTest = Xtest*pa;
        tempP = 0;
        s = S{i+1};
        u = U{i+1};
        p = C{i+1};
        for k = 1:K
            sk = s(:,k);
            X_mu = tempTest - u(:,k)';
            tempP = tempP+p(k)/sqrt(prod(sk))*exp((-0.5*sum(((X_mu.^2)'.*(1./sk))',2)));
        end
        tempP = tempP*P(i+1);
        resP(:,i+1) = tempP;
    end


    [~, y_prediction] = max(resP,[],2);
    y_prediction = y_prediction-1;
    error_rate = sum(y_prediction ~= ytest)/length(ytest)
end

