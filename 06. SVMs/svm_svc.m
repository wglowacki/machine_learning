close all
clearvars

%% Generate data.

rng(1); % For reproducibility

% Generate data from a normal distribution.
n_cls = 20; % Number of samples in each class.
X = vertcat(...
    horzcat(normrnd(0.5,1, n_cls,1), normrnd(0.4,1, n_cls,1)), ...
    horzcat(normrnd(-0.3,1, n_cls,1), normrnd(-0.5,1, n_cls,1)) ...
    );
Y = vertcat(-1 * ones(n_cls,1), +1 * ones(n_cls,1));

x1_boundary = [min(X(:,1)) - 0.1, max(X(:,1)) + 0.1];

%% Fit model.

SVMModel = fitcsvm(X,Y, 'ClassNames',[-1,1]);

% f(x) = (X / s)' * Beta + b = 0 => x2 = -(x1 * beta1 + b * s) / beta2
f_dec_x2 = @(x1) -(x1 * SVMModel.Beta(1) + SVMModel.Bias * SVMModel.KernelParameters.Scale) / SVMModel.Beta(2);

figure(1);
clf(1);
h = [];

h(1) = plot(X(Y == -1,1), X(Y == -1,2), 'r.', 'MarkerSize',15, 'DisplayName','Class -1');
hold on
h(2) = plot(X(Y == +1,1), X(Y == +1,2), 'b.', 'MarkerSize',15, 'DisplayName','Class +1');
h(3) = plot(X(SVMModel.IsSupportVector,1),X(SVMModel.IsSupportVector,2),'ko', 'MarkerSize',10, 'DisplayName','Support vector');
h(4) = plot(x1_boundary, f_dec_x2(x1_boundary), 'k--', 'DisplayName','Decision boundary');
axis equal
xlabel('X_1'); ylabel('X_2')
xlim(x1_boundary)
hold off
title('Data')
legend(h, 'Location','northeastoutside')

%% Make predictions.

newX = [
    1, -0.4;
    1, -0.85
    ];

[label,score] = predict(SVMModel,newX);

hold on
h(5) = plot(newX(label == -1,1), newX(label == -1,2), 'ro', 'MarkerSize',8, 'DisplayName','Class -1 (predicted)');
h(6) = plot(newX(label == +1,1), newX(label == +1,2), 'bo', 'MarkerSize',8, 'DisplayName','Class +1 (predicted)');
hold off
legend(h, 'Location','northeastoutside')

%% Get posteriors.

% Get posterior probabilities for predictions.
% REF: https://www.mathworks.com/help/stats/compactclassificationsvm.fitposterior.html#bt72een
ScoreSVMModel = fitPosterior(SVMModel, X, Y);
[~, postProbs] = predict(ScoreSVMModel, newX);
