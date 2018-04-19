close all
clearvars

%%

load fisheriris
X = meas;
Y = species;

figure(1); clf(1);
gscatter(X(:,1), X(:,2), Y, 'rgb', 'osd');
xlabel('Sepal length');
ylabel('Sepal width');

%%

X_cols = 1:2;

rng(1); % For reproducibility

t = templateSVM('KernelFunction','gaussian');

% Mdl = fitcecoc(X(:,1:2), Y, 'FitPosterior',1);
Mdl = fitcecoc(X(:,X_cols), Y, 'Learners',t, 'FitPosterior',1);

Mdl.ClassNames
CodingMat = Mdl.CodingMatrix

isLoss = resubLoss(Mdl)
% The classification error is small, but the classifier might have been
% overfit. You can cross-validate the classifier using crossval.

%%

[label, ~, ~, Posterior] = predict(Mdl, X(:,X_cols));

%%

xMin = min(X(:,X_cols));
xMax = max(X(:,X_cols));

d = 0.1;
[x1Grid, x2Grid] = meshgrid(xMin(1):d:xMax(1), xMin(2):d:xMax(2));
[~, ~, ~, PosteriorRegions] = predict(Mdl, horzcat(x1Grid(:), x2Grid(:)));

%%

figure(2); clf(2);
h = [];

gscatter(X(:,1), X(:,2), Y, 'rgb', 'osd');
xlabel('Sepal length');
ylabel('Sepal width');

hold on
contour(x1Grid, x2Grid, reshape(PosteriorRegions(:,1),size(x1Grid)), [0.5, 0.5]);
contour(x1Grid, x2Grid, reshape(PosteriorRegions(:,2),size(x1Grid)), [0.5, 0.5]);
contour(x1Grid, x2Grid, reshape(PosteriorRegions(:,3),size(x1Grid)), [0.5, 0.5]);
hold off
