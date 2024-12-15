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

% ...existing code...

% 2. Data Processing and Variance Analysis
num_users = userRange_max - userRange_min + 1;
num_features = size(userData(1).trainFeatures, 2);
num_samples = size(userData(1).trainFeatures, 1);

% Create 3D array [users x samples x features]
all_data = zeros(num_users, num_samples, num_features);
for user = userRange_min:userRange_max
    all_data(user,:,:) = userData(user).trainFeatures;
end

% Calculate mean across samples for each user
user_means = squeeze(mean(all_data, 2));  % [users x features]

% Calculate inter-user variance for each feature
inter_user_variance = var(user_means, 0, 1);  % [1 x features]

% 3. Visualization
% Create output directory if it doesn't exist
output_dir = '3d_analysis';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Calculate variance for each user across their samples
user_variances = zeros(num_users, num_features);
for user = 1:num_users
    user_variances(user,:) = var(squeeze(all_data(user,:,:)), 0, 1);
end

% 3.1 3D Surface plot of feature variances across users
figure('Name', 'User Feature Variances');
[X, Y] = meshgrid(1:num_features, 1:num_users);
surf(X, Y, user_variances);
xlabel('Feature Index');
ylabel('User Index');
zlabel('Variance');
title('Feature Variances Across Users');
colormap(jet);
colorbar;
view(45, 30);

% Save the surface plot
saveas(gcf, fullfile(output_dir, 'feature_variances_surface.fig'));
saveas(gcf, fullfile(output_dir, 'feature_variances_surface.png'));

% 3.2 3D Bar plot of feature variances across users
figure('Name', 'User Feature Variances - Bar Plot');
bar3(user_variances);
xlabel('Feature Index');
ylabel('User Index');
zlabel('Variance');
title('Feature Variances by User');
colormap(hot);
colorbar;
view(45, 30);

% Save the bar plot
saveas(gcf, fullfile(output_dir, 'feature_variances_bar.fig'));
saveas(gcf, fullfile(output_dir, 'feature_variances_bar.png'));

% Optional: Add mean variance line plot for comparison
figure('Name', 'Mean Feature Variances');
plot(mean(user_variances, 1), 'LineWidth', 2);
xlabel('Feature Index');
ylabel('Mean Variance');
title('Average Feature Variance Across All Users');
grid on;

% Save the line plot
saveas(gcf, fullfile(output_dir, 'mean_feature_variances.fig'));
saveas(gcf, fullfile(output_dir, 'mean_feature_variances.png'));
