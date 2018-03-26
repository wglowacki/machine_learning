
%   TOPIC: Data analysis
%
% ------------------------------------------------------------------------

close all; clear all;
clearvars

smarket = readtable('./data/smarket.csv');
smarket.Direction = categorical(smarket.Direction, {'Up','Down'});
smarket_subset = smarket(:, [2:7, 9]);
glm = fitglm(smarket_subset);

yhat = predict(glm, smarket) > 0.5;  % predict() ma w³asny system zamiany zmiennych jakoœciowych na zmienne zerojedynkowe...
yhat = categorical(yhat, [0,1], {'Up', 'Down'});
 
% W UCI nalezy skozystac z funkcji confusionmat() i recznie obliczyc "error rate"
% cp = classperf(cellstr(smarket.Direction), cellstr(yhat));
cp = confusionmat(cellstr(smarket.Direction), cellstr(yhat));
cp_err = (cp(1,2) + cp(2,1))/sum(sum(cp));

is_train = (smarket.Year < 2005);
smarket_train = smarket(is_train,:);
smarket_test = smarket(~is_train,:);
modelspec = 'Direction~Lag1+Lag2';
train_model = fitglm(smarket_train,modelspec,'Distribution','binomial');
train_hat = predict(train_model, smarket_test) > 0.5;  % predict() ma w³asny system zamiany zmiennych jakoœciowych na zmienne zerojedynkowe...
train_hat = categorical(train_hat, [0,1], {'Up', 'Down'});

cp_train = confusionmat(cellstr(smarket_test.Direction), cellstr(train_hat));
cp_err_train = (cp_train(1,2) + cp_train(2,1))/sum(sum(cp_train));