function [w] = binaryLR(X,y,lambda,tol,maxiter)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%Let g define the gradient vector
%Let h define the hassian matrix

    switch nargin
        case 4
            maxiter = 10000;
        case 3
            maxiter = 10000;
            tol = 10^-4;
    end
    
    [rowX,colX] = size(X);
    %rowX is the number of samples
    %col is the number of features
    w = rand(colX,1).*0.00001;
    %w = rand(colX,1);
    %do w'*x'
    %w is d*1
    

    k = 0;
    error = inf;
    
    while error>tol && k<maxiter 
        tic
        e = exp(-y.*(w'*X')');
        p = 1./(1+e);
        %e is n by 1
        %p is n by 1 too, mothafaka
        
        indP = find(y==1);
        %indP contains indices of possitive y
        indN = find(y==-1);
        %indP contains indices of negative y
        
        g_pre = -p.*((1./p)-1).*y.*X;
        g = 1./(length(indP)).*sum(g_pre(indP,:))'+1./(length(indN)).*sum(g_pre(indN,:))'+2.*lambda.*w;
        
        %[a,b] = size(g)
        % g is d by 1
        % d here is 3072
        % g done
       
        
        p_k = (p-p.^2).*X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Calculating the Hassian Matrix             
        p_k(indP,:) = (1./(length(indP))).*p_k(indP,:);
        p_k(indN,:) = (1./(length(indN))).*p_k(indN,:);
        h = p_k'*X;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        dia = 2*lambda*eye(colX);
        h = h+dia;
        w_new = w - h\g;

        error = sqrt(mean((w_new - w).^2))
        w = w_new;
        k = k+1
        toc
    end
    

end

