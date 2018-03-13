%
%   TOPIC: Linear Regression - Qualitative Predictors
%
% ------------------------------------------------------------------------

close all; clearvars; clc;

seats = readtable('./data/carseats.csv');
lm = fitlm(seats, 'Sales~CompPrice+Income+Advertising+Population+Price+ShelveLoc+Age+Education+Urban+US+Income:Advertising+Price:Age');
display(lm);
p1 = anova(lm)