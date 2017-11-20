function [c,u,S] = GMM(X,K)
% model includes ??RK+ and for each 1?k?K, ?k ?Rd and Sk ?Sd+
% ?k ? 0,  Kk=1 ?k = 1, Sk symmetric and positive definite.
% random initialization suffices for full credit.
% alternatively, can initialize r by randomly 
% assigning each data to one of the K components
% here we use c to replace pi

% n is the number of data points
% d is the number of features
% K is the number of gaussian models
% u has dimension [d,K]


    maxiter = 500;

    [n,d] = size(X);
    tol = 10.^-5;

    S = rand(d,K)+1;
    % S = repmat(diag(X'*X),1,K);
    % initialize the covraiance matrix
    % each column is a covariance matrix
    % when obtaining the covariance matrix
    % use diag(S(:,i)), this creates a diagonal matrix
    
    u = sum(X)'./n;
    u = repmat(u,1,K)+0.05.*max(u).*rand(d,K);
    % initialize the mean matrix
    % each column is a mean vector
    % use u(:,i) to obtain the mean vector for ith cluster
    % when calculating (X-u) do (X-u(:,k)')
    
    c = rand(K,1);
    c = c./sum(c);
    % initialize 
    

    iter = 1;
    while iter < maxiter
        
        
        for k = 1:K
            %r(:,k) = c(k).*(prod(S(:,k)).^(-0.5)).*exp(-0.5.*sum(((X-u(:,k)').^2).*(1./S(:,k)'),2));            
            temp = log(c(k))-0.5.*sum(log(S(:,k)));
            r(:,k) = temp-0.5*sum(((X-u(:,k)').^2).*(1./S(:,k)'),2);
        end
        
        %disp('Size of r');
        %[a,b] = size(r);
        %r has dimension n by k
        %here, rik' = log(rik) 
            
        
        r = exp(r);
        R = sum(r,2);
        r = r./R;
        %normalization of r
        
        l(iter) = -sum(log(R));
        %calculating the loss               
        
        if iter > 1 && abs(l(iter)-l(iter-1)) <= tol*abs(l(iter))
        %The break condition
            break;
        end
        
    
        R_k = sum(r)';
        % R_k is a column vector
        % R_k has dimension k by 1
        c = R_k./n;
        
        for k = 1:K
            u(:,k) = (sum(X.*r(:,k))./R_k(k))';
        end
        %updating mean values
        
        
        for k = 1:K
            S(:,k) = sum(r(:,k).*(X.^2))'./R_k(k) - u(:,k).^2;
            %r(:,k) is n by 1
            %X.^2 is n by d
        end
        %updating covariance matrices
        
        S = S + eps * (S < 1e-10 * eps);
        %preventing the covariance from being too small
        
        iter = iter+1;    
    end
    
end

