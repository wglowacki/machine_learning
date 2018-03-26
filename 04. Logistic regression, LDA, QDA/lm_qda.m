%
%   TOPIC: Quadratic Discriminant Analysis
%
% ------------------------------------------------------------------------

close all; clc
clearvars

smarket = readtable('./data/smarket.csv');
%smarket.Direction = categorical(smarket.Direction, {'Up','Down'});
is_train = (smarket.Year < 2005);
smarket_train = smarket(is_train,:);
smarket_test = smarket(~is_train,:);

gscatter(smarket.Lag1,smarket.Lag2,smarket.Direction, 'rb','ox',[],'off');
hold on;
qda_mdl = fitcdiscr(smarket_train, 'Direction~Lag1+Lag2', 'DiscrimType', 'quadratic');

fprintf('Class names:\n')
disp(qda_mdl.ClassNames)
fprintf('Group means:\n')
disp(qda_mdl.Mu)
fprintf('Prior probabilities of groups:\n')
disp(qda_mdl.Prior)

qda_mdl.ClassNames([1 2])
K = qda_mdl.Coeffs(1,2).Const;
L = qda_mdl.Coeffs(1,2).Linear;
Q = qda_mdl.Coeffs(1,2).Quadratic;

f = @(x1,x2) K + L(1)*x1 + L(2)*x2 + Q(1,1)*x1.^2 + ...
    (Q(1,2)+Q(2,1))*x1.*x2 + Q(2,2)*x2.^2;
h2 = ezplot(f,[-6 6 -6 6]);
h2.Color = 'g';
h2.LineWidth = 2;

[~, score, ~] = predict(qda_mdl, smarket_test);
yhat_t90 = (score(:,2) > 0.50);
yhat_t90 = categorical(yhat_t90, [0,1], {'Up', 'Down'});

cp_train = confusionmat(cellstr(smarket_test.Direction), cellstr(yhat_t90));
cp_err_train = (cp_train(1,2) + cp_train(2,1))/sum(sum(cp_train));
