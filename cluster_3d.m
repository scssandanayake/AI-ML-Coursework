clear all;
close all;
clc;

rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;



% 1. Data Loading and Preprocessing
% Define file patterns for each user
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest = 'Acc_TimeD_FreqD_MDay';

% First: Load and combine features for each user separately
fprintf('Loading data for each user...\n');

% Initialize storage datasets
userData = struct();

% Load data for each user
for user = userRange_min:userRange_max
  userStr = sprintf('U%02d', user);

  % Load training and test data
  trainFile = [ userStr '_' filePatternsTrain '.mat'];
  testFile = [ userStr '_' filePatternsTest '.mat'];

  if exist(trainFile, 'file') && exist(testFile, 'file')
    trainData = load(trainFile);
    testData = load(testFile);

    userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
    userData(user).testFeatures = testData.(char(fieldnames(testData)));
  else
    fprintf('Missing data files for user %d\n', user);
  end
end

% 2. Feature Processing and Visualization
fprintf('Processing features for visualization...\n');

% Combine all training features into one matrix with debug info
allFeatures = [];
userLabels = [];
for user = userRange_min:userRange_max
    fprintf('Processing user %d...\n', user);
    if isfield(userData, num2str(user))
        features = userData(user).trainFeatures;
        fprintf('Found %d features for user %d\n', size(features, 1), user);
        if ~isempty(features)
            allFeatures = [allFeatures; features];
            userLabels = [userLabels; ones(size(features, 1), 1) * user];
        end
    else
        % Direct array access instead of isfield
        if ~isempty(userData(user).trainFeatures)
            features = userData(user).trainFeatures;
            fprintf('Found %d features for user %d (direct access)\n', size(features, 1), user);
            allFeatures = [allFeatures; features];
            userLabels = [userLabels; ones(size(features, 1), 1) * user];
        end
    end
end

% Debug information
fprintf('Total features collected: %d\n', size(allFeatures, 1));
fprintf('Feature dimension: %d\n', size(allFeatures, 2));

% Check if we have enough features
if isempty(allFeatures)
    error('No features found in the data. Check data loading process.');
end

% Apply PCA with dimension checking
[coeff, score, latent] = pca(allFeatures);
numComponents = size(score, 2);  % Get actual number of components

% Create figure
figure('Position', [100 100 800 600]);

% Plot based on available dimensions
if numComponents >= 3
    scatter3(score(:,1), score(:,2), score(:,3), 50, userLabels, 'filled');
    zlabel('Third Principal Component');
    rotate3d on;
elseif numComponents == 2
    scatter(score(:,1), score(:,2), 50, userLabels, 'filled');
elseif numComponents == 1
    scatter(score(:,1), zeros(size(score,1),1), 50, userLabels, 'filled');
else
    error('Insufficient data for PCA visualization');
end

colormap(jet(userRange_max));
colorbar;
xlabel('First Principal Component');
ylabel('Second Principal Component');
title(sprintf('User Feature Clusters in PCA Space (%d components)', numComponents));
grid on;

% Save the plot
saveas(gcf, '3d_cluster.fig', 'fig');
saveas(gcf, '3d_cluster.png', 'png');

% Calculate and display explained variance for available components
explained = latent / sum(latent) * 100;
fprintf('Number of PCA components available: %d\n', numComponents);
fprintf('Total explained variance: %.2f%%\n', sum(explained));


