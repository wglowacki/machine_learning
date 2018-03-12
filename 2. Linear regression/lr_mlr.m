%
%   TOPIC: Multiple Linear Regression
%
% ------------------------------------------------------------------------

close all
clearvars

boston = readtable('./data/boston.csv');
boston_subset = boston(:, {'MEDV', 'LSTAT', 'AGE'});
scatter3(boston_subset.MEDV, boston_subset.LSTAT, boston_subset.AGE, 'filled')
hold on
lm = fitlm(boston_subset);
display(lm)

x1 = boston_subset.AGE;
x2 = boston_subset.LSTAT;
y = boston_subset.MEDV;
X = [ones(size(x1)) x1 x2 x1.*x2];
b = regress(y,X)

x1fit = min(x1):1:max(x1);
x2fit = min(x2):1:max(x2);
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
YFIT = b(1)+ b(2)*X1FIT + b(3)*X2FIT + b(4)*X1FIT.*X2FIT;
mesh(X1FIT,X2FIT,YFIT)