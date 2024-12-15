%% FeedForwardNet with Overfitting Prevention %%%%%

% Complete User Authentication using MLP Neural Network - Binary Classification
clear all;
close all;
clc;

rng(100);  % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;

% Overfitting prevention parameters
TrainTargetImposterRatio = 1/5;  % Fixed ratio 1:5
dropoutRate = 0.3;  % Dropout rate for regularization
l2RegParam = 1e-4;  % L2 regularization parameter
performanceGoal = 1e-5;  % Performance goal for training
minGrad = 1e-6;  % Minimum gradient for training
earlyStoppingPatience = 10;  % Patience for early stopping
maxEpochs = 500;  % Maximum number of training epochs
learningRate = 0.01; % Learning rate

numUsers = userRange_max - userRange_min + 1;

% Feed forward net architecture
trainFcn = 'trainscg';
hiddenLayerSizes = [131];
hiddenLayerActivationFcns = {'logsig'};
outputLayerActivationFcn = 'tansig';
performanceFcn = 'crossentropy';

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
  trainFile = ['dataset/' userStr '_' filePatternsTrain '.mat'];
  testFile = ['dataset/' userStr '_' filePatternsTest '.mat'];

  if exist(trainFile, 'file') && exist(testFile, 'file')
    trainData = load(trainFile);
    testData = load(testFile);

    userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
    userData(user).testFeatures = testData.(char(fieldnames(testData)));
  else
    fprintf('Missing data files for user %d\n', user);
  end
end

% Find minimum number of samples across users for training and testing
minSamplesTrain = inf;
minSamplesTest = inf;
for user = userRange_min:userRange_max
  if ~isempty(userData(user).trainFeatures)
    minSamplesTrain = min(minSamplesTrain, size(userData(user).trainFeatures, 1));
  end
  if ~isempty(userData(user).testFeatures)
    minSamplesTest = min(minSamplesTest, size(userData(user).testFeatures, 1));
  end
end

% Random
% Leave-Out Users list generation
% leaveOutUsersList = zeros(1, numUsers); % Initialize the list

% for targetUser = userRange_min:userRange_max
%   % Generate a random number between 1 and 10 excluding the targetUser
%   options = setdiff(1:10, targetUser); % Exclude targetUser
%   leaveOutUsersList(targetUser) = options(randi(length(options))); % Select a random number
% end
leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

% Train one model for each user (one-vs-all approach)
models = cell(numUsers, 1);

% Initialize results storage
userMetrics = zeros(numUsers, 14);
userPerformance = zeros(numUsers, 3); % For timing, memory, throughput
userSimilarityData = cell(3, numUsers, numUsers);

for targetUser = userRange_min:userRange_max
  % Prepare Training set
  trainTargetSampleCount = size(userData(targetUser).trainFeatures, 1);
  trainImposterSampleCount = trainTargetSampleCount*(1/TrainTargetImposterRatio);
  trainSamplesPerImposter = floor(trainImposterSampleCount/(numUsers-1));

  XTrain = [userData(targetUser).trainFeatures];
  yTrain = ones(trainTargetSampleCount, 1);
  trainImposterFeatures = [];
  trainImposterLabels = [];
  for imposterUser = 1:numUsers
    if imposterUser ~= targetUser && imposterUser ~= leaveOutUsersList(targetUser)
      selectedIdx = randperm(size(userData(imposterUser).trainFeatures, 1), trainSamplesPerImposter);
      trainImposterFeatures = [trainImposterFeatures; userData(imposterUser).trainFeatures(selectedIdx, :)];
      trainImposterLabels = [trainImposterLabels; zeros(trainSamplesPerImposter, 1)];
    end
  end

  XTrain = [XTrain; trainImposterFeatures];
  yTrain = [yTrain; trainImposterLabels];

  % Verify the train classes are balanced to the given ratio
  assert(sum(yTrain == 1) == trainTargetSampleCount);
  assert(sum(yTrain == 0) == trainImposterSampleCount-trainSamplesPerImposter);

  % Prepare Testing set
  testTargetSampleCount = size(userData(targetUser).testFeatures, 1);
  testImposterSampleCount = testTargetSampleCount*(numUsers-1);
  testSamplesPerImposter = floor(testImposterSampleCount/(numUsers-1));

  XTest = [userData(targetUser).testFeatures];
  yTest = ones(testTargetSampleCount, 1);

  testImposterFeatures = [];
  testImposterLabels = [];
  testUserLabels = ones(testTargetSampleCount, 1) * targetUser;

  for imposterUser = 1:numUsers
    if imposterUser ~= targetUser
      selectedIdx = randperm(size(userData(imposterUser).testFeatures, 1), testSamplesPerImposter);
      testImposterFeatures = [testImposterFeatures; userData(imposterUser).testFeatures(selectedIdx, :)];
      testImposterLabels = [testImposterLabels; zeros(testSamplesPerImposter, 1)];
      testUserLabels = [testUserLabels; ones(testSamplesPerImposter, 1) * imposterUser];
    end
  end

  XTest = [XTest; testImposterFeatures];
  yTest = [yTest; testImposterLabels];

  % Verify the test classes are balanced
  assert(sum(yTest == 1) == testTargetSampleCount);
  assert(sum(yTest == 0) == testImposterSampleCount);

  % Create and configure the network
  net = feedforwardnet(131, 'trainscg');
  net.userdata.note = "Initial Feedforward Neural Network with selected Leave-Out Users";
  net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
  net.userdata.dropoutRate = dropoutRate;
  net.userdata.l2RegParam = l2RegParam;
  net.userdata.performanceGoal = performanceGoal;
  net.userdata.minGrad = minGrad;
  net.userdata.earlyStoppingPatience = earlyStoppingPatience;
  net.userdata.maxEpochs = maxEpochs;
  net.userdata.learningRate = learningRate;
  net.userdata.targetUser = sprintf('User %d', targetUser);
  net.performFcn = performanceFcn; % Performance function

  % Configure layers
  for layerNo = 1:length(hiddenLayerActivationFcns)
    net.layers{layerNo}.transferFcn = hiddenLayerActivationFcns{layerNo};
  end
  net.layers{end}.transferFcn = outputLayerActivationFcn; % set Output Layer Activation Function

  % Configure training parameters
  net.trainParam.epochs = maxEpochs;
  net.trainParam.goal = performanceGoal;
  net.trainParam.min_grad = minGrad;
  net.performParam.regularization = l2RegParam;
  net.trainParam.max_fail = earlyStoppingPatience;
  net.trainParam.lr = learningRate;

  % Train the network
  tic;
  [net, tr] = train(net, XTrain', yTrain');
  trainTime = toc;

  % Store the model
  models{targetUser} = net;

  % Measure memory usage
  modelInfo = whos('net');
  memoryUsage = modelInfo.bytes / (1024^2); % Convert to MB

  % Evaluate Network & Calculate metrics
  tic;
  yPred = net(XTest')';
  inferenceTime = toc;
  throughput = size(XTest, 1) / inferenceTime; % samples/second

  % Store performance metrics
  userPerformance(targetUser, :) = [trainTime + inferenceTime, memoryUsage, throughput];
  % Store Similarities
  modelUserSimilarities = [testUserLabels, yPred];

  yPred = yPred > 0.5;
  yPred = double(yPred);

  tp = sum(yPred & yTest);
  fp = sum(yPred & ~yTest);
  fn = sum(~yPred & yTest);
  tn = sum(~yPred & ~yTest);

  precision = tp/(tp + fp);
  recall = tp/(tp + fn);
  f1_score = 2 * (precision * recall)/(precision + recall);
  specificity = tn/(tn + fp);
  accuracy = (tp + tn)/(tp + tn + fp + fn);

  mcc = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

  [X,Y,T,AUC] = perfcurve(yTest, yPred, true);

  % Calculate FAR and FRR at different thresholds
  FAR = X;
  FRR = 1 - Y;

  % Find the threshold where FAR and FRR are equal (EER)
  [~, eerIdx] = min(abs(FAR - FRR));
  EER = (FAR(eerIdx) + FRR(eerIdx)) / 2;

  % Calculate & Store Similarity stats
  % Precompute statistics matrices
  similarity_means = zeros(1, numUsers);
  similarity_mids = zeros(1, numUsers);
  similarity_mid_variations = zeros(1, numUsers);

  for user = userRange_min:userRange_max
    indices = find(modelUserSimilarities(:, 1) == user);
    similarity_means(1, user) = mean(modelUserSimilarities(indices, 2));
    similarity_min = min(modelUserSimilarities(indices, 2));
    similarity_max = max(modelUserSimilarities(indices, 2));
    similarity_mids(1, user) = (similarity_max + similarity_min)/2;
    similarity_mid_variations(1, user) = similarity_max - similarity_mids(user);
  end

  % Store similarity data
  userSimilarityData(1, targetUser, :) = num2cell(similarity_means);
  userSimilarityData(2, targetUser, :) = num2cell(similarity_mids);
  userSimilarityData(3, targetUser, :) = num2cell(similarity_mid_variations);

  % Store metrics
  userMetrics(targetUser, :) = [accuracy, precision, recall, specificity, f1_score, mcc, ...
    FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, AUC, ...
    size(XTrain, 1), trainTargetSampleCount, trainImposterSampleCount, ...
    size(XTest, 1)];

  % Display comprehensive results for each user
  fprintf('\n==== Individual User Performance ====\n');
  fprintf('\nUser %d Results:\n', targetUser);
  fprintf('Accuracy: %.2f%%\n', userMetrics(targetUser, 1)*100);
  fprintf('Precision: %.2f%%\n', userMetrics(targetUser, 2)*100);
  fprintf('Recall: %.2f%%\n', userMetrics(targetUser, 3)*100);
  fprintf('Specificity: %.2f%%\n', userMetrics(targetUser, 4)*100);
  fprintf('F1-Score: %.2f%%\n', userMetrics(targetUser, 5)*100);
  fprintf('Matthews Correlation Coefficient: %.4f\n', userMetrics(targetUser, 6));
  fprintf('False Acceptance Rate: %.2f%%\n', userMetrics(targetUser, 7));
  fprintf('False Rejection Rate: %.2f%%\n', userMetrics(targetUser, 8));
  fprintf('Equal Error Rate: %.2f%%\n', userMetrics(targetUser, 9));
  fprintf('AUC Score: %.4f\n', userMetrics(targetUser, 10));
  fprintf('Training Time: %.4f seconds\n', userPerformance(targetUser, 1));
  fprintf('Memory Usage: %.2f MB\n', userPerformance(targetUser, 2));
  fprintf('Throughput: %.2f samples/second\n', userPerformance(targetUser, 3));

  % Plot confusion matrix
  figure;
  plotconfusion(yTest', yPred');
  title(sprintf('Confusion Matrix - User %d', targetUser));

  % Plot ROC curve
  figure;
  plot(X,Y);

  hold on;

  % Mark the EER point on the ROC curve
  plot(FAR(eerIdx), Y(eerIdx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
  text(FAR(eerIdx), Y(eerIdx), sprintf('  EER = %.2f%%', EER * 100), 'VerticalAlignment', 'bottom');

  % Add labels and title
  xlabel('False Positive Rate');
  ylabel('True Positive Rate');
  title(sprintf('ROC Curve - User %d (AUC = %.3f)', targetUser, AUC));
  grid on;
  hold off;
end

% Compute average metrics
avgMetrics = mean(userMetrics, 1);
avgPerformance = mean(userPerformance, 1);

% Create comprehensive results structure
results = struct(...
  'Ratio', '1:6', ...
  'AvgAccuracy', avgMetrics(1)*100, ...
  'AvgPrecision', avgMetrics(2)*100, ...
  'AvgRecall', avgMetrics(3)*100, ...
  'AvgSpecificity', avgMetrics(4)*100, ...
  'AvgF1Score', avgMetrics(5)*100, ...
  'AvgMCC', avgMetrics(6), ...
  'AvgFAR', avgMetrics(7), ...
  'AvgFRR', avgMetrics(8), ...
  'AvgEER', avgMetrics(9), ...
  'AvgAUC', avgMetrics(10), ...
  'AvgTrainingSetSize', avgMetrics(11), ...
  'AvgTrainTargetSamples', avgMetrics(12), ...
  'AvgTrainImposterSamples', avgMetrics(13), ...
  'AvgTestSetSize', avgMetrics(14), ...
  'AvgTotalTime', avgPerformance(1), ...
  'AvgMemoryUsage', avgPerformance(2), ...
  'AvgThroughput', avgPerformance(3));

% Format and display neural network details
fprintf('\n==== Neural Network Architecture ====\n');
fprintf('Input Layer: %d neurons\n', size(XTrain, 2));
for i = 1:length(hiddenLayerSizes)
  fprintf('Hidden Layer %d: %d neurons (%s)\n', i, hiddenLayerSizes(i), hiddenLayerActivationFcns{i});
end
fprintf(['Output Layer: ' net.outputs{end}.size ' neuron (' outputLayerActivationFcn ')\n']);
fprintf(['Training Algorithm: ' trainFcn '\n']);
fprintf('Performance Function: Cross-Entropy\n');
fprintf('L2 Regularization: %e\n', l2RegParam);
fprintf('Max Epochs: %d\n', maxEpochs);

% Display performance benchmarks
fprintf('\n==== Performance Benchmarks ====\n');
fprintf('Average Training Time: %.4f seconds (±%.4f)\n', mean(userPerformance(:,1)), std(userPerformance(:,1)));
fprintf('Average Memory Usage: %.2f MB (±%.2f)\n', mean(userPerformance(:,2)), std(userPerformance(:,2)));
fprintf('Average Throughput: %.2f samples/second (±%.2f)\n', mean(userPerformance(:,3)), std(userPerformance(:,3)));

% Create and display summary table
summaryTable = table((1:numUsers)', ...
  userPerformance(:,1), ...
  userPerformance(:,2), ...
  userPerformance(:,3), ...
  userMetrics(:,1)*100, ...
  userMetrics(:,2)*100, ...
  userMetrics(:,3)*100, ...
  userMetrics(:,4)*100, ...
  userMetrics(:,5)*100, ...
  userMetrics(:,6), ...
  userMetrics(:,7), ...
  userMetrics(:,8), ...
  userMetrics(:,9), ...
  userMetrics(:,10), ...
  'VariableNames', {...
  'User', 'InferenceTime_sec', 'MemoryUsage_MB', 'Throughput_samples_per_sec', ...
  'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1_Score', ...
  'MCC', 'FAR', 'FRR', 'EER', 'AUC'});

% Compute overall metrics
overallMetrics = table(mean(userPerformance(:,1)), ...
  mean(userPerformance(:,2)), ...
  mean(userPerformance(:,3)), ...
  mean(userMetrics(:,1)*100), ...
  mean(userMetrics(:,2)*100), ...
  mean(userMetrics(:,3)*100), ...
  mean(userMetrics(:,4)*100), ...
  mean(userMetrics(:,5)*100), ...
  mean(userMetrics(:,6)), ...
  mean(userMetrics(:,7)), ...
  mean(userMetrics(:,8)), ...
  mean(userMetrics(:,9)), ...
  mean(userMetrics(:,10)), ...
  'VariableNames', {...
  'Avg_InferenceTime_sec', 'Avg_MemoryUsage_MB', 'Avg_Throughput_samples_per_sec', ...
  'Avg_Accuracy', 'Avg_Precision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
  'Avg_MCC', 'Avg_FAR', 'Avg_FRR', 'Avg_EER', 'Avg_AUC'});

fprintf('\n==== Summary Table ====\n');
disp(summaryTable);
disp('Overall Metrics:');
disp(overallMetrics);

% Create similarity matrix and format strings
similarityMatrix = zeros(numUsers, numUsers);
for i = 1:numUsers
  for j = 1:numUsers
    val = cell2mat(userSimilarityData(1,i,j));
    mid = cell2mat(userSimilarityData(2,i,j));
    var = cell2mat(userSimilarityData(3,i,j));
    similarityMatrix(i,j) = val;
    labelStrings{i,j} = sprintf('%.2f\nM: %.2f\n(±%.3f)', val, mid, var);
  end
end

% Create figure and plot heatmap
figure('Position', [100 100 800 600]);
imagesc(similarityMatrix);

% Use a light colormap
colormap(sky); % Or try: bone, pink, summer
c = colorbar;
c.Label.String = 'Similarity Score';

% Add text annotations
[X,Y] = meshgrid(1:numUsers, 1:numUsers);
for i = userRange_min:userRange_max
  for j = userRange_min:userRange_max
    text(i, j, labelStrings{j,i}, ...
      'HorizontalAlignment', 'center', ...
      'Color', 'black', ...
      'FontSize', 10);

    if leaveOutUsersList(i) == j
      hold on;
      plot(j, i, 'rs', 'MarkerSize', 60);
      hold off;
    end
  end
end

% Customize axes
set(gca, 'XTick', 1:numUsers, 'XTickLabel', userRange_min:userRange_max);
set(gca, 'YTick', 1:numUsers, 'YTickLabel', userRange_min:userRange_max);
xlabel("User N's similarity score");
ylabel("User N's Model");
title('User similarity scores for each user model');
axis square;

% Save the results
save('benchmark_results.mat', 'summaryTable', 'overallMetrics');

% Save the models
save('user_authentication_models.mat', 'models');
