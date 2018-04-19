close all
clearvars

%% Generate data.

rng(1); % For reproducibility

n_cls = 100; % Number of samples in each class.

r = sqrt(rand(n_cls,1)); % Radius
t = 2 * pi * rand(n_cls,1);  % Angle
X_cls1 = [r .* cos(t), r .* sin(t)]; % Points

r2 = sqrt(3 * rand(n_cls,1)+1); % Radius
t2 = 2 * pi * rand(n_cls,1);      % Angle
X_cls2 = [r2 .* cos(t2), r2 .* sin(t2)]; % points

X = vertcat(X_cls1, X_cls2);
Y = vertcat(-1 * ones(n_cls,1), +1 * ones(n_cls,1));

%% Fit a model.

SVMModel = fitcsvm(X,Y, 'KernelFunction','rbf', ...
    'BoxConstraint',Inf, 'ClassNames',[-1,1]);

% Predict scores over the grid
d = 0.02;
[x1Grid, x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)), ...
    min(X(:,2)):d:max(X(:,2)));
xGrid = horzcat(x1Grid(:), x2Grid(:));
[~, scores] = predict(SVMModel, xGrid);

figure(1);
clf(1);
h = [];

h(1) = plot(X(Y == -1,1), X(Y == -1,2), 'r.', 'MarkerSize',15, 'DisplayName','Class -1');
hold on
h(2) = plot(X(Y == +1,1), X(Y == +1,2), 'b.', 'MarkerSize',15, 'DisplayName','Class +1');
ezpolar(@(x) 1); ezpolar(@(x) 2);
h(3) = plot(X(SVMModel.IsSupportVector,1),X(SVMModel.IsSupportVector,2),'ko', 'MarkerSize',10, 'DisplayName','Support vector');
[~, h(4)] = contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k', 'DisplayName','Decision boundary');

xlabel('X_1'); ylabel('X_2')
axis equal
hold off
title('Data')
legend(h, 'Location','northeastoutside')
