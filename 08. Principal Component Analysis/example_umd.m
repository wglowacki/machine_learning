clear all; close all; clc;

load irisdata.txt           % array of size 150 x 5

%% Part I
X = irisdata(:,1:4)';       % 150 columns of length 4
spec = irisdata(:,5)';      % vector of length 150 with species (0,1,or 2)
n = size(X,2);

figure(1)                   % plot x_i vs x_j for all combinations of i,j
for i=1:4
  for j=1:i-1
     subplot(3,3,(i-1)+3*(j-1))
     scatter(X(i,:),X(j,:),7,spec,'filled')
     xlabel(sprintf('x%g',i)); ylabel(sprintf('x%g',j))
  end
end

%% Part II
Xmean = mean(X,2)           % find mean
A = X - Xmean*ones(1,n);    % subtract mean from each point
rho = norm(A,'fro')^2       % total variation of data

[U,S,V] = svd(A,'econ');    % find singular value decomposition
sigma = diag(S)             % singular values

rho = norm(sigma)^2         % gives same variation as above


%% Part III
C = S(1:3,1:3)*V(:,1:3)';   % first 3 coefficients for each point, same as U(:,1:3)'*A;

q2 = norm(sigma(1:2))^2/rho % part of variation captured by first 2 components
figure(2);
scatter(C(1,:),C(2,:),17,spec,'filled')
xlabel('PC1'); ylabel('PC2')
title(sprintf('2 components, captures %.4g%% of total variation',100*q2))

%% Part IV

q3 = norm(sigma(1:3))^2/rho   % part of variation captured by first 3 components
figure(3);
scatter3(C(1,:),C(2,:),C(3,:),27,spec,'filled')
xlabel('PC1'); ylabel('PC2'); zlabel('PC3')
title(sprintf('3 components, captures %.4g%% of total variation',100*q3))
% you can rotate and spin this graph with the mouse