%
%   TOPIC: Simple Linear Regression
%
% ------------------------------------------------------------------------

%W tym zadaniu stworzysz model regresji liniowej przewiduj?cej
%?redni? warto?? domu na podstawie odsetka gospodarstw domowych

%% Simple linear regression
close all; clearvars;

boston = readtable('./data/boston.csv');
boston_subset = boston(:, {'MEDV', 'LSTAT'});
%scatter(boston_subset{:,1}, boston_subset{:,2})
scatter(boston_subset.MEDV, boston_subset.LSTAT)
lm = fitlm(boston_subset, 'MEDV~LSTAT');
disp(lm)

coofficiency = coefCI(lm)
hold on
plot(lm)
a = linspace(5,50);
plot(a, coofficiency(2,1) * a + coofficiency(1,1));
plot(a, coofficiency(2,2) * a + coofficiency(1,2));

%% Normal probability plot of residuals
figure(2)
plotResiduals(lm, 'probability');

%% Model with nonlinaer regression
figure(3)
scatter(boston_subset.MEDV, boston_subset.LSTAT)
new_lm = fitlm(boston_subset, 'MEDV~LSTAT+LSTAT^2');
disp(new_lm)
plot(new_lm)
hold off

%% Compare both models
p1 = anova(lm)
p2 = anova(new_lm)
