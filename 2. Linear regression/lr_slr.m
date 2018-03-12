%
%   TOPIC: Simple Linear Regression
%
% ------------------------------------------------------------------------

close all
clearvars

boston = readtable('./data/boston.csv');
boston_subset = boston(:, {'MEDV', 'LSTAT'});
%scatter(boston_subset{:,1}, boston_subset{:,2})
scatter(boston_subset.MEDV, boston_subset.LSTAT)
lm = fitlm(boston_subset, 'linear');
disp(lm)

coofficiency = coefCI(lm)
hold on
plot(lm)
a = linspace(5,50);
plot(a, coofficiency(2,1) * a + coofficiency(1,1));
plot(a, coofficiency(2,2) * a + coofficiency(1,2));
hold off

plotResiduals(lm, 'probability')