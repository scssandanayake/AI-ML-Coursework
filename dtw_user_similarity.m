folder_path = 'dataset';

% TIME DOMAIN

% Initializing the array for merged data in TD
all_data = [];

% Looping through each user
for user_idx = 1:10
    % Loading data for the current user from two file patterns
    file1 = sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user_idx);
    file2 = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user_idx);

    % Full file paths
    file1_path = fullfile(folder_path, file1);
    file2_path = fullfile(folder_path, file2);

    % Loading the data from both files
    data1 = load(file1_path);
    data2 = load(file2_path);

    % Extracting the feature matrices (36 x 88 from each file)
    feature_data1 = data1.(char(fieldnames(data1)));
    feature_data2 = data2.(char(fieldnames(data2)));

    % Concatenating the data for the specific user (72 x 88)
    user_data = [feature_data1; feature_data2];

    % Appending the user data to the all_data_TD matrix
    all_data = [all_data; user_data];
end

% Normalizing user data (z-score normalization)
mean_vals = mean(all_data, 1);    % Computing the mean of each feature (across all users)
std_vals = std(all_data, 0, 1);   % Computing the standard deviation of each feature (across all users)

% Handling cases where standard deviation is zero
std_vals(std_vals == 0) = 1;

% Performing z-score normalization
all_data_normalized = (all_data - mean_vals) ./ std_vals;

% Number of users and samples per user
num_users = 10;
samples_per_user = 72;
num_features = size(all_data_normalized, 2);  % Number of features (columns)

% Initializing DTW distance matrix
dtw_matrix = zeros(num_users, num_users);

% Looping through all pairs of users and calculating DTW (Dynamic Time Warping) distances
for user_idx_1 = 1:num_users
    for user_idx_2 = user_idx_1:num_users  % Avoiding duplicate comparisons
        % Extracting data for User 1
        start_idx_1 = (user_idx_1 - 1) * samples_per_user + 1;
        end_idx_1 = user_idx_1 * samples_per_user;
        user_data_1 = reshape(all_data_normalized(start_idx_1:end_idx_1, :), [], 1);

        % Extracting data for User 2
        start_idx_2 = (user_idx_2 - 1) * samples_per_user + 1;
        end_idx_2 = user_idx_2 * samples_per_user;
        user_data_2 = reshape(all_data_normalized(start_idx_2:end_idx_2, :), [], 1);

        % Calculate DTW distance
        [dist, ~, ~] = dtw(user_data_1, user_data_2);

        % Storing the DTW distance in the matrix
        dtw_matrix(user_idx_1, user_idx_2) = dist;
        dtw_matrix(user_idx_2, user_idx_1) = dist;  % Symmetric matrix

        % Displaying the DTW distance
        fprintf('DTW distance between User %d and User %d: %.4f\n', user_idx_1, user_idx_2, dist);
    end
end

% Plotting the DTW distance matrix as a heatmap
figure('Position', [100 100 800 600]);
imagesc(dtw_matrix); % Displaying the matrix as a color image
colorbar;            % Adding a colorbar to show the scale
title('DTW for TD Distance Matrix');
xlabel('User Index');
ylabel('User Index');
set(gca, 'XTick', 1:num_users, 'YTick', 1:num_users); % Labeling the axes with user indices

% Initializing a cell array to store the results
most_similar_users_TD = cell(num_users, 1);

for user_idx = 1:num_users
    % Extracting the DTW distances for the current user
    dtw_row = dtw_matrix(user_idx, :);

    % Setting diagonal element to Inf to exclude self-comparison
    dtw_row(user_idx) = Inf;

    % Finding the index of the minimum DTW distance
    [min_dist, similar_user_idx] = min(dtw_row);

    % Storing the results
    most_similar_users_TD{user_idx} = struct('User', user_idx, ...
        'MostSimilarUser', similar_user_idx, ...
        'Distance', min_dist);
end

[X,Y] = meshgrid(1:num_users, 1:num_users);
for targetUser = 1:num_users
    for secondaryUser = 1:num_users
        % Format the DTW value as a string with 2 decimal places
        dtw_value = sprintf('%.2f', dtw_matrix(targetUser, secondaryUser));

        % Use proper syntax for text function with x,y coordinates and string value
        text(secondaryUser, targetUser, dtw_value, ...
            'HorizontalAlignment', 'center', ...
            'Color', 'black', ...
            'FontSize', 8);

        if most_similar_users_TD{targetUser}.MostSimilarUser == secondaryUser
            hold on;
            plot(secondaryUser, targetUser, 'rs', 'MarkerSize', 60);
            hold off;
        end
    end
end

% Displaying results
fprintf('Most Similar Users:\n');
for user_idx = 1:num_users
    fprintf('User %d is most similar to User %d with DTW distance: %.4f\n', ...
        most_similar_users_TD{user_idx}.User, ...
        most_similar_users_TD{user_idx}.MostSimilarUser, ...
        most_similar_users_TD{user_idx}.Distance);
end