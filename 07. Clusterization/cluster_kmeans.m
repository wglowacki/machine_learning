%
%   TOPIC: K-Means Clustering
%
% ------------------------------------------------------------------------

close all
clear clc
clearvars

%% Load data.

I = imread('hestain.png');

figure(1);
imshow(I), title('H&E image (original)');

%% Perform k-means clustering.
nColors = 3;
rows = size(I(:,:,1), 1); 
cols = size(I(:,:,1), 2);
r_vect = double(reshape(I(:,:,1), rows*cols, 1));
g_vect = double(reshape(I(:,:,2), rows*cols, 1));
b_vect = double(reshape(I(:,:,3), rows*cols, 1));
rgb = [r_vect, g_vect, b_vect];
k_meas_cluster = kmeans(rgb, nColors, 'Replicates', 3);

%% Show (image) labeling.

figure(2); clf(2);
pixel_labels = reshape(k_meas_cluster, rows, cols);
imshow(pixel_labels, []), title('image labeled by cluster index');

%% Show data in each cluster.

segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = I;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

figure(3); clf(3)
subplot(2,2,1); imshow(I); title('original image')
subplot(2,2,2); imshow(segmented_images{1}); title('objects in cluster 1');
subplot(2,2,3); imshow(segmented_images{2}); title('objects in cluster 2');
subplot(2,2,4); imshow(segmented_images{3}); title('objects in cluster 3');

%% Show clustering in RGB color space.

seg1 = segmented_images(1);
seg2 = segmented_images(2);
segmented_images(3);


figure(4); clf(4)
colors = 'rgb';
markers = '...';
for idx = 1:3
    plot3(segmented_images{idx}(:,:,1), segmented_images{idx}(:,:,2), segmented_images{idx}(:,:,3), strcat(colors(idx), markers(idx)))
    hold on;
end
hold off
title('K-means clustering')
xlabel('R'); ylabel('G'); zlabel('B')
grid