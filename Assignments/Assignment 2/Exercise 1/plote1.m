%load('10X_[1,2,3,4,5]_euclidean.mat')
%load('10X_[1,5,10,20,50]_euclidean.mat')
%load('5X_[1,5,10,20,50]_manhattan.mat')
load('5X_[1,5,10,20,50]_chebyshev.mat')
plot(k_set,validation_k_error,'o');
title('5CV[1,5,10,20,50]chebyshev');
xlabel('k');
ylabel('percentage of mistakes');
[y x] = min(validation_k_error);
hold on;
plot(x,y,'r');