clear all;
close all;
clc;

rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;

filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest = 'Acc_TimeD_FreqD_MDay';

% First: Load and combine features for each user separately
fprintf('Loading data for each user...\n');

% Initialize storage datasets
userData = struct();

% Load data for each user
for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);
    trainFile = ['dataset/' userStr '_' filePatternsTrain '.mat'];
    testFile = ['dataset/' userStr '_' filePatternsTest '.mat'];

    if exist(trainFile, 'file') && exist(testFile, 'file')
        trainData = load(trainFile);
        testData = load(testFile);

        % Store the data directly in numbered fields
        userData.(sprintf('user%d', user)).fday = trainData.(char(fieldnames(trainData)));
        userData.(sprintf('user%d', user)).mday = testData.(char(fieldnames(testData)));
        fprintf('Loaded data for user %d\n', user);
    else
        fprintf('Missing data files for user %d\n', user);
    end
end

% Analyze and plot similarity
fprintf('Analyzing all sample pairs similarities between FDay and MDay...\n');

for user = userRange_min:userRange_max
    userField = sprintf('user%d', user);
    if isfield(userData, userField)
        fday_data = userData.(userField).fday;
        mday_data = userData.(userField).mday;

        % Calculate similarity matrix between all samples
        n_samples = size(fday_data, 1);
        similarity_matrix = zeros(n_samples, n_samples);

        % Calculate cosine similarity between all sample pairs
        for i = 1:n_samples
            for j = 1:n_samples
                similarity_matrix(i,j) = dot(fday_data(i,:), mday_data(j,:)) / ...
                    (norm(fday_data(i,:)) * norm(mday_data(j,:)));
            end
        end

        % Create figure with adjusted size
        figure('Position', [100 100 1200 500]);

        % 3D Scatter plot with spheres
        subplot(1,2,1);
        hold on;
        for i = 1:n_samples
            for j = 1:n_samples
                % Only plot points with similarity above threshold for clarity
                if similarity_matrix(i,j) > 0.5  % Adjust threshold as needed
                    scatter3(i, j, similarity_matrix(i,j), ...
                        similarity_matrix(i,j)*200, ... % Size based on similarity
                        similarity_matrix(i,j), ...    % Color based on similarity
                        'filled', ...
                        'MarkerFaceAlpha', 0.6);      % Add transparency
                end
            end
        end
        hold off;

        % Adjust 3D plot appearance
        colormap('jet');
        colorbar;
        title(sprintf('User %02d: Sample Similarity (Spheres)', user));
        xlabel('First Day Samples');
        ylabel('Second Day Samples');
        zlabel('Cosine Similarity');
        view(45, 45);
        grid on;

        % Keep the existing heatmap in second subplot
        subplot(1,2,2);
        imagesc(similarity_matrix);
        colormap('jet');
        colorbar;
        title(sprintf('User %02d: Sample Similarity Heatmap', user));
        xlabel('First Day Samples');
        ylabel('Second Day Samples');
        axis square;

        % Print statistics
        diagonal_sim = diag(similarity_matrix);
        fprintf('User %02d - Diagonal Mean: %.4f, Overall Mean: %.4f\n', ...
            user, mean(diagonal_sim), mean(similarity_matrix(:)));
    else
        fprintf('No data structure for user %02d\n', user);
    end
end