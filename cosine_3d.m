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
userData = struct('trainFeatures', [], 'testFeatures', []);
userData = repmat(userData, 1, userRange_max);

% Load data for each user
for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);
    
    % Load training and test data
    trainFile = [userStr '_' filePatternsTrain '.mat'];
    testFile = [userStr '_' filePatternsTest '.mat'];
    
    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData = load(testFile);
        
        % Debug print
        fprintf('Loading data for user %d\n', user);
        fprintf('Train file: %s\n', trainFile);
        fprintf('Test file: %s\n', testFile);
        
        % Store data in structure
        userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
        userData(user).testFeatures = testData.(char(fieldnames(testData)));
        
        % Verify data loading
        [r, c] = size(userData(user).trainFeatures);
        fprintf('Loaded data dimensions for user %d: %d samples x %d features\n', user, r, c);
    else
        fprintf('Missing data files for user %d\n', user);
        fprintf('Tried to load:\n');
        fprintf('Train: %s\n', trainFile);
        fprintf('Test: %s\n', testFile);
    end
end

% 2. Visualization
for user = userRange_min:userRange_max
    % Skip if no data
    if isempty(userData(user).trainFeatures) || isempty(userData(user).testFeatures)
        fprintf('Skipping user %d - no data\n', user);
        continue;
    end
    
    % Create new figure for each user
    figure('Name', sprintf('User %d Point-wise Similarity', user));
    
    % Get data
    fday_data = userData(user).trainFeatures;  % 36x131
    mday_data = userData(user).testFeatures;   % 36x131
    
    % Calculate point-wise cosine similarity for each feature
    [numSamples, numFeatures] = size(fday_data);
    similarities = zeros(numSamples, numSamples);
    
    for i = 1:numSamples
        for j = 1:numSamples
            % Get the corresponding points
            point_fday = fday_data(i,:);
            point_mday = mday_data(j,:);
            
            % Calculate cosine similarity
            similarities(i,j) = dot(point_fday, point_mday) / ...
                (norm(point_fday) * norm(point_mday));
        end
    end
    
    % Create meshgrid for 3D plotting
    [X, Y] = meshgrid(1:numSamples, 1:numSamples);
    
    % Plot
    scatter3(X(:), Y(:), similarities(:), 50, similarities(:), 'filled');
    
    % Customize plot
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Cosine Similarity';
    caxis([0 1]);  % Set color limits
    
    xlabel('FDay Sample Index');
    ylabel('MDay Sample Index');
    zlabel('Cosine Similarity');
    title(sprintf('User %d: Point-wise Similarity\nMean Similarity = %.3f', ...
        user, mean(similarities(:))));
    
    % Add grid and enable rotation
    grid on;
    rotate3d on;
    
    % Set view angle and axis limits
    view(45, 30);
    xlim([1 numSamples]);
    ylim([1 numSamples]);
    zlim([0 1]);
end

fprintf('\nVisualization complete!\n');