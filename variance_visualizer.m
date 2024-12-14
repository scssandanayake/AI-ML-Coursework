folder_path = 'CW-Data';

% TIME DOMAIN

% Initializing the array for merged data in TD
all_data_TD = [];

% Looping through each user
for user_idx = 1:10
    % Loading data for the current user from two file patterns
    file1 = sprintf('U%02d_Acc_TimeD_FDay.mat', user_idx);
    file2 = sprintf('U%02d_Acc_TimeD_MDay.mat', user_idx);

    % Full file paths
    file1_path = fullfile(folder_path, file1);
    file2_path = fullfile(folder_path, file2);

    % Loading the data from both files
    data1 = load(file1_path);
    data2 = load(file2_path);

    % Extracting the feature matrices (36 x 88 from each file)
    feature_data1 = data1.Acc_TD_Feat_Vec;
    feature_data2 = data2.Acc_TD_Feat_Vec;

    % Concatenating the data for the specific user (72 x 88)
    user_data_TD = [feature_data1; feature_data2];

    % Appending the user data to the all_data_TD matrix
    all_data_TD = [all_data_TD; user_data_TD];
end

% Number of users and samples per users
num_users = 10;
samples_per_user = 72;
num_features_TD = size(all_data_TD, 2);  % Number of features (columns)

% Initializing arrays to store the statistical values
mean_vals_TD_per_user = zeros(num_users, num_features_TD);  % Mean values of users for each feature
std_vals_TD_per_user = zeros(num_users, num_features_TD);   % Standard deviation values of users for each feature
intra_variance_TD_per_user = zeros(num_users, num_features_TD);  % Intra-user variance values of users for each feature
inter_variance_TD = zeros(1, num_features_TD);  % Inter-user variance values of users for each feature

% Calculation Of Mean, Standard Deviation, Intra-Variance per user
for user_idx = 1:num_users
    start_idx = (user_idx - 1) * samples_per_user + 1;
    end_idx = user_idx * samples_per_user;

    % Extracting user data for the specific feature
    user_data_TD = all_data_TD(start_idx:end_idx, :);

    % Calculating the mean and standard deviation for each user
    mean_vals_TD_per_user(user_idx, :) = mean(user_data_TD);
    std_vals_TD_per_user(user_idx, :) = std(user_data_TD);

    % Intra-user variance (variance within a single user's samples for the specific feature)
    intra_variance_TD_per_user(user_idx, :) = var(user_data_TD);
end

% Inter-Variance Calculation
% Calculate the inter-user variance (variance of the user means)
inter_variance_TD = var(mean_vals_TD_per_user);

fprintf('Mean values for each feature in TD (per user): \n');
disp(mean_vals_TD_per_user);

fprintf('Standard deviation values for each feature in TD (per user): \n');
disp(std_vals_TD_per_user);

fprintf('Intra-variance values for each feature in TD (per user): \n');
disp(intra_variance_TD_per_user);

fprintf('Inter-variance values for each feature in TD (across users): \n');
disp(inter_variance_TD);

% Mean, Standard Deviation and Intra-Variance across all users
mean_vals_TD = mean(mean_vals_TD_per_user, 1);
std_vals_TD = mean(std_vals_TD_per_user, 1);
intra_variance_TD_mean = mean(intra_variance_TD_per_user, 1);  % Mean intra-variance across all users

fprintf('Mean values for each feature in TD (across all users): \n');
disp(mean_vals_TD);

fprintf('Standard deviation values for each feature in TD (across all users): \n');
disp(std_vals_TD);

fprintf('Mean intra-variance values for each feature in TD (across all users): \n');
disp(intra_variance_TD_mean);

% FREQUENCY DOMAIN

% Initializing the array for merged data in FD
all_data_FD = [];

% Looping through each user
for user_idx = 1:10
    % Loading data for the current user from two file patterns
    file3 = sprintf('U%02d_Acc_FreqD_FDay.mat', user_idx);
    file4 = sprintf('U%02d_Acc_FreqD_MDay.mat', user_idx);

    % Full file paths
    file3_path = fullfile(folder_path, file3);
    file4_path = fullfile(folder_path, file4);

    % Loading the data from both files
    data3 = load(file3_path);
    data4 = load(file4_path);

    % Extractinng the feature matrices (36 x 43 from each file)
    feature_data3 = data3.Acc_FD_Feat_Vec;
    feature_data4 = data4.Acc_FD_Feat_Vec;

    % Concatenating the data for the specific user (72 x 43)
    user_data_FD = [feature_data3; feature_data4];

    % Append the user data to the all_data matrix
    all_data_FD = [all_data_FD; user_data_FD];
end

% Number of features (columns)
num_features_FD = size(all_data_FD, 2);

% Initializing arrays to store the statistical values
mean_vals_FD_per_user = zeros(num_users, num_features_FD);  % Mean values of users for each feature
std_vals_FD_per_user = zeros(num_users, num_features_FD);   % Standard deviation values of users for each feature
intra_variance_FD_per_user = zeros(num_users, num_features_FD);  % Intra-user variance values of users for each feature
inter_variance_FD = zeros(1, num_features_FD);  % Inter-user variance values of users for each feature
user_means_FD = zeros(num_users, num_features_FD);
% Calculation Of Mean, Standard Deviation, Intra-Variance per user
for user_idx = 1:num_users
    start_idx = (user_idx - 1) * samples_per_user + 1;
    end_idx = user_idx * samples_per_user;

    % Extracting user data for the specific feature
    user_data_FD = all_data_FD(start_idx:end_idx, :);

    % Calculating the mean and standard deviation for each user
    mean_vals_FD_per_user(user_idx, :) = mean(user_data_FD);
    std_vals_FD_per_user(user_idx, :) = std(user_data_FD);

    % Intra-user variance (variance within a single user's samples for the specific feature)
    intra_variance_FD_per_user(user_idx, :) = var(user_data_FD);
end

% Inter-Variance Calculation
% Calculate the inter-user variance (variance of the user means)
inter_variance_FD = var(mean_vals_FD_per_user);

fprintf('Mean values for each feature in FD (per user): \n');
disp(mean_vals_FD_per_user);

fprintf('Standard deviation values for each feature in FD (per user): \n');
disp(std_vals_FD_per_user);

fprintf('Intra-variance values for each feature in FD (per user): \n');
disp(intra_variance_FD_per_user);

fprintf('Inter-variance values for each feature in FD (across users): \n');
disp(inter_variance_FD);

% Mean, Standard Deviation and Intra-Variance across all users
mean_vals_FD = mean(mean_vals_FD_per_user, 1);
std_vals_FD = mean(std_vals_FD_per_user, 1);
intra_variance_FD_mean = mean(intra_variance_FD_per_user, 1);  % Mean intra-variance across all users

fprintf('Mean values for each feature in FD (across all users): \n');
disp(mean_vals_FD);

fprintf('Standard deviation values for each feature in FD (across all users): \n');
disp(std_vals_FD);

fprintf('Mean intra-variance values for each feature in FD (across all users): \n');
disp(intra_variance_FD_mean);

% TIME DOMAIN AND FREQUENCY DOMAIN

% Initializing the array for merged data in TDFD
all_data_TDFD = [];

% Looping through each user
for user_idx = 1:10
    % Loading data for the current user from two file patterns
    file5 = sprintf('U%02d_Acc_TimeD_FreqD_FDay.mat', user_idx);
    file6 = sprintf('U%02d_Acc_TimeD_FreqD_MDay.mat', user_idx);

    % Full file paths
    file5_path = fullfile(folder_path, file5);
    file6_path = fullfile(folder_path, file6);

    % Loading the data from both files
    data5 = load(file5_path);
    data6 = load(file6_path);

    % Extractinng the feature matrices (36 x 131 from each file)
    feature_data5 = data5.Acc_TDFD_Feat_Vec;
    feature_data6 = data6.Acc_TDFD_Feat_Vec;

    % Concatenating the data for the specific user (72 x 131)
    user_data_TDFD = [feature_data5; feature_data6];

    % Append the user data to the all_data matrix
    all_data_TDFD = [all_data_TDFD; user_data_TDFD];
end

% Number of features (columns)
num_features_TDFD = size(all_data_TDFD, 2);

% Initializing arrays to store the statistical values
mean_vals_TDFD_per_user = zeros(num_users, num_features_TDFD);  % Mean values of users for each feature
std_vals_TDFD_per_user = zeros(num_users, num_features_TDFD);   % Standard deviation values of users for each feature
intra_variance_TDFD_per_user = zeros(num_users, num_features_TDFD);  % Intra-user variance values of users for each feature
inter_variance_TDFD = zeros(1, num_features_TDFD);  % Inter-user variance values of users for each feature

% Calculation Of Mean, Standard Deviation, Intra-Variance per user
for user_idx = 1:num_users
    start_idx = (user_idx - 1) * samples_per_user + 1;
    end_idx = user_idx * samples_per_user;

    % Extracting user data for the specific feature
    user_data_TDFD = all_data_TDFD(start_idx:end_idx, :);

    % Calculating the mean and standard deviation for each user
    mean_vals_TDFD_per_user(user_idx, :) = mean(user_data_TDFD);
    std_vals_TDFD_per_user(user_idx, :) = std(user_data_TDFD);

    % Intra-user variance (variance within a single user's samples for the specific feature)
    intra_variance_TDFD_per_user(user_idx, :) = var(user_data_TDFD);
end

% Inter-Variance Calculation
% Calculate the inter-user variance (variance of the user means)
inter_variance_TDFD = var(mean_vals_TDFD_per_user);

fprintf('Mean values for each feature in TDFD (per user): \n');
disp(mean_vals_TDFD_per_user);

fprintf('Standard deviation values for each feature in TDFD (per user): \n');
disp(std_vals_TDFD_per_user);

fprintf('Intra-variance values for each feature in TDFD (per user): \n');
disp(intra_variance_TDFD_per_user);

fprintf('Inter-variance values for each feature in TDFD (across users): \n');
disp(inter_variance_TDFD);

% Mean, Standard Deviation and Intra-Variance across all users
mean_vals_TDFD = mean(mean_vals_TDFD_per_user, 1);
std_vals_TDFD = mean(std_vals_TDFD_per_user, 1);
intra_variance_TDFD_mean = mean(intra_variance_TDFD_per_user, 1);  % Mean intra-variance across all users

fprintf('Mean values for each feature in TDFD (across all users): \n');
disp(mean_vals_TDFD);

fprintf('Standard deviation values for each feature in TDFD (across all users): \n');
disp(std_vals_TDFD);

fprintf('Mean intra-variance values for each feature in TDFD (across all users): \n');
disp(intra_variance_TDFD_mean);

% Create figure for Time Domain
figure('Name', 'Time Domain Analysis');

% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_TD_per_user', 'LineWidth', 1.5);
plot(intra_variance_TD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance Distribution Across Users with Mean (TD)');
xlabel('Feature Index');
ylabel('Variance');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_TD_per_user', 'LineWidth', 1.5);
plot(std_vals_TD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation Distribution Across Users with Mean (TD)');
xlabel('Feature Index');
ylabel('Standard Deviation');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_TD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (TD)');
xlabel('Feature Index');
ylabel('Variance');
grid on;

% Create figure for Frequency Domain
figure('Name', 'Frequency Domain Analysis');

% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_FD_per_user', 'LineWidth', 1.5);
plot(intra_variance_FD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance Distribution Across Users with Mean (FD)');
xlabel('Feature Index');
ylabel('Variance');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_FD_per_user', 'LineWidth', 1.5);
plot(std_vals_FD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation Distribution Across Users with Mean (FD)');
xlabel('Feature Index');
ylabel('Standard Deviation');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_FD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (FD)');
xlabel('Feature Index');
ylabel('Variance');
grid on;

% Create figure for Time-Frequency Domain
figure('Name', 'Time-Frequency Domain Analysis');

% Subplot 1: Combined intra-variances and mean
subplot(3,1,1);
hold on;
plot(intra_variance_TDFD_per_user', 'LineWidth', 1.5);
plot(intra_variance_TDFD_mean, 'k--', 'LineWidth', 2);
hold off;
title('Intra-variance Distribution Across Users with Mean (TDFD)');
xlabel('Feature Index');
ylabel('Variance');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 2: Standard deviation with mean
subplot(3,1,2);
hold on;
plot(std_vals_TDFD_per_user', 'LineWidth', 1.5);
plot(std_vals_TDFD, 'k--', 'LineWidth', 2);
hold off;
title('Standard Deviation Distribution Across Users with Mean (TDFD)');
xlabel('Feature Index');
ylabel('Standard Deviation');
legend({'User 1', 'User 2', 'User 3', 'User 4', 'User 5', ...
    'User 6', 'User 7', 'User 8', 'User 9', 'User 10', 'Mean'});
grid on;

% Subplot 3: Inter-variance only
subplot(3,1,3);
plot(inter_variance_TDFD, 'r-', 'LineWidth', 1.5);
title('Inter-variance (TDFD)');
xlabel('Feature Index');
ylabel('Variance');
grid on;
