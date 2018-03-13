%
%   TOPIC: Multiple Linear Regression
%
% ------------------------------------------------------------------------

%% Multi linear regression
close all; clearvars; clc;

boston = readtable('./data/boston.csv');
boston_subset = boston(:, {'LSTAT', 'AGE', 'MEDV'});
scatter3(boston_subset.LSTAT, boston_subset.AGE, boston_subset.MEDV, 'filled')
hold on
lm = fitlm(boston_subset, 'MEDV~AGE+LSTAT');
display(lm)
x1 = boston_subset.LSTAT;
x2 = boston_subset.AGE;
y = boston_subset.MEDV;
X = [ones(size(x1)), x1, x2];
b = regress(y,X) % returns a p-by-1 vector b of coefficient estimates for 
%a multilinear regression of the responses in y on the predictors in X.
%X is an n-by-p matrix of p predictors at each of n observations.
%Yis an n-by-1 vector of observed responses.

x1fit = min(x1):1:max(x1);
x2fit = min(x2):1:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1)+ b(2)*X1FIT + b(3)*X2FIT;
mesh(X1FIT,X2FIT,YFIT);

%% new model of simple linear regression

new_lm = fitlm(boston_subset, 'MEDV~AGE+LSTAT+AGE:LSTAT');
display(new_lm)
X = [ones(size(x1)), x1, x2, x1.*x2];
b = regress(y,X);
YFIT = b(1)+ b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
surf(X1FIT,X2FIT,YFIT);
hold off





