function [alpha] = KbinaryLR(X,y,lambda,kernel,epsilon,tol,maxiter)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

%Let g define the gradient vector
%Let h define the hassian matrix

    switch nargin
        case 6
            maxiter = 10;
        case 5
            maxiter = 10;
            tol = 10^-4;
    end
    
    [rowX,colX] = size(X);
    %rowX is the number of samples
    %col is the number of features
    
    switch kernel
        case 'linear'
            K = X*X';
        case 'polynomial'
            K = (1+X*X').^5;
        case 'gaussian'
            %{
            D = sqdist(X',X').^2;
            K = exp(-D./epsilon);
            %}
            K = kmatrix(X,X,'gaussian',epsilon);
    end
    
    
    
    alpha = zeros(rowX,1);
    %initializing alpha
    
    k = 0;
    error = inf;
    
    while error>tol && k<maxiter
        tic
        %e = exp(-y.*(w'*X')');
        e = exp(-y.*(alpha'*K')');
        %e = exp(-y.*(K'*alpha));
        p = 1./(1+e);
        %e is n by 1
        %p is n by 1 too, mothafaka
        
        indP = find(y==1);
        %indP contains indices of possitive y
        indN = find(y==-1);
        %indP contains indices of negative y
        
        %g_pre = -p.*((1./p)-1).*y.*X;
        g_pre = -p.*((1./p)-1).*y.*K;
        %g = 1./(length(indP)).*sum(g_pre(indP,:))'+1./(length(indN)).*sum(g_pre(indN,:))'+2.*lambda.*w;
        g = 1./(length(indP)).*sum(g_pre(indP,:))'+1./(length(indN)).*sum(g_pre(indN,:))'+2.*lambda.*(K*alpha);

        % [a,b] = size(g)
        % g is d by 1
        % d here is 1950
        % g done
       
        
        %p_k = (p-p.^2).*X;
        p_k = (p-p.^2).*K;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Calculating the Hassian Matrix             
        p_k(indP,:) = (1./(length(indP))).*p_k(indP,:);
        p_k(indN,:) = (1./(length(indN))).*p_k(indN,:);
        %h = p_k'*X;
        h = p_k'*K;
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %dia = 2*lambda*eye(colX);
        dia = 2.*lambda.*K;
        h = h+dia;
        
        alpha_new = alpha - h\g;
        error = sqrt(mean((alpha_new - alpha).^2))
        alpha = alpha_new;
        k = k+1
        toc
    end
end
