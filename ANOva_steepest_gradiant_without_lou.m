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
%leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

% Feature Selection Parameters
anovaThreshold = 0.05;  % Threshold for ANOVA p-values
topFeaturePercent = 0.75;  % Top 75% features to select

% Initialize storage for selected features
selectedFeatures = cell(numUsers, 1);

% Initialize storage for feature analysis
featureAnalysis = struct();

for targetUser = userRange_min:userRange_max
  % Prepare data for feature selection
  X = userData(targetUser).trainFeatures;
  y = ones(size(X, 1), 1);
  for imposterUser = 1:numUsers
    if imposterUser ~= targetUser 
      X = [X; userData(imposterUser).trainFeatures];
      y = [y; zeros(size(userData(imposterUser).trainFeatures, 1), 1)];
    end
  end

  % Get total number of features
  numFeatures = size(X, 2);
  
  % ANOVA Feature Selection
  pValues = zeros(1, numFeatures);
  for i = 1:numFeatures
    pValues(i) = anova1(X(:, i), y, 'off');
  end
  anovaSelected = find(pValues < anovaThreshold);
  
  % Mutual Information Feature Selection
  miScores = zeros(1, numFeatures);
  for i = 1:numFeatures
    miScores(i) = calculate_mutual_information(X(:, i), y);
  end
  [~, miRanking] = sort(miScores, 'descend');
  miSelected = miRanking(1:round(topFeaturePercent*numFeatures));

  % Steepest Gradient Feature Selection
  net = feedforwardnet(10, 'trainscg');
  net = train(net, X', y');
  gradients = abs(net.IW{1});
  [~, sgRanking] = sort(mean(gradients, 1), 'descend');
  sgSelected = sgRanking(1:round(topFeaturePercent*numFeatures));

  
  % Plot Feature Analysis
  figure('Name', sprintf('Feature Analysis - User %d', targetUser));
  
  % Plot 1: Feature Importance Scores with proper normalization
  subplot(2,2,1);
  
  % Initialize feature scores matrix
  numFeatures = size(X, 2);
  featureScores = zeros(numFeatures, 3);
  
  % Process ANOVA scores (lower p-value = higher importance)
  featureScores(:,1) = 1 - normalize(reshape(pValues, [], 1), 'range');
  
  % Process MI scores
  featureScores(:,2) = normalize(reshape(miScores, [], 1), 'range');
  
  % Process gradient scores
  meanGradients = mean(gradients, 1)';
  if length(meanGradients) ~= numFeatures
      % Interpolate gradient scores if dimensions don't match
      meanGradients = interp1(1:length(meanGradients), meanGradients, linspace(1, length(meanGradients), numFeatures));
  end
  featureScores(:,3) = normalize(meanGradients, 'range');
  
  % Plot stacked bar chart
  bar(featureScores, 'stacked');
  title(sprintf('Feature Importance by Method (%d features)', numFeatures));
  legend('ANOVA', 'MI', 'Gradient');
  xlabel('Feature Index');
  ylabel('Normalized Importance');
  
  % Calculate weighted combined scores
  weights = [0.4, 0.3, 0.3]; % Weights for ANOVA, MI, and Gradient
  combinedScores = featureScores * weights';
  [sortedScores, sortedIdx] = sort(combinedScores, 'descend');
  
  % Update selected features
  selectedFeatureIdx = sortedIdx(1:round(topFeaturePercent*numFeatures));
  
  % Plot 2: Correlation Matrix
  subplot(2,2,2);
  correlationMatrix = corr(X(:, selectedFeatureIdx));
  imagesc(correlationMatrix);
  colormap(jet);
  colorbar;
  title(sprintf('Feature Correlation Matrix\n(%d features)', length(selectedFeatureIdx)));
  
  % Plot 3: Box Plots for Top Features
  subplot(2,2,3);
  topN = min(5, length(selectedFeatureIdx));
  topFeatures = selectedFeatureIdx(1:topN);
  
  % Prepare data for boxplot
  boxData = [];
  groupLabels = {};
  for i = 1:topN
      featureValues = X(:, topFeatures(i));
      boxData = [boxData; featureValues];
      groupLabels = [groupLabels; repmat({sprintf('Feature %d', i)}, length(featureValues), 1)];
  end
  
  % Create grouped boxplot
  boxplot(boxData, groupLabels);
  hold on;
  % Add color coding for genuine/impostor
  scatter(find(y==1), boxData(y==1), 10, 'b', '.');
  scatter(find(y==0), boxData(y==0), 10, 'r', '.');
  hold off;
  
  title(sprintf('Top %d Features Distribution', topN));
  xlabel('Feature Index');
  ylabel('Feature Value');
  legend('Genuine', 'Impostor', 'Location', 'eastoutside');
  grid on;
  
  % Plot 4: Method Contribution
  subplot(2,2,4);
  methodContribution = sum(featureScores, 1);
  if all(isfinite(methodContribution))
      pie(methodContribution);
      title('Feature Selection Method Contribution');
      legend({'ANOVA', 'MI', 'Gradient'}, 'Location', 'eastoutside');
  else
      warning('Non-finite values found in method contribution, skipping pie chart.');
  end

  % Additional visualization: Feature importance trend
  figure('Name', sprintf('Feature Importance Trends - User %d', targetUser));
  
  % Plot combined importance scores
  subplot(2,1,1);
  combinedScores = mean(featureScores, 2);
  [sortedScores, sortedIdx] = sort(combinedScores, 'descend');
  bar(sortedScores(1:min(20,end)));
  title('Top 20 Features by Combined Importance');
  xlabel('Feature Rank');
  ylabel('Importance Score');
  grid on;
  
  % Plot individual method contributions for top features
  subplot(2,1,2);
  topK = min(20, length(sortedIdx));
  topFeatureScores = featureScores(sortedIdx(1:topK), :);
  bar(topFeatureScores, 'grouped');
  title('Method Contributions for Top Features');
  xlabel('Feature Rank');
  ylabel('Score');
  legend('ANOVA', 'MI', 'Gradient');
  grid on;

  featureAnalysis(targetUser).scores = featureScores;
  featureAnalysis(targetUser).combinedScores = combinedScores;
  featureAnalysis(targetUser).selectedFeatures = selectedFeatureIdx;
  featureAnalysis(targetUser).correlationMatrix = correlationMatrix;
  

  fprintf('\nFeature Selection Summary for User %d:\n', targetUser);
  fprintf('Top 5 features: %s\n', mat2str(selectedFeatureIdx(1:min(5,end))));
  fprintf('Average correlation: %.4f\n', mean(abs(correlationMatrix(triu(true(size(correlationMatrix)),1)))));
  fprintf('Number of selected features: %d\n', length(selectedFeatureIdx));
  
  % Store final selected feature indices
  selectedFeatures{targetUser} = selectedFeatureIdx;
  
  % Print selected features and their counts
  fprintf('\nFeature Selection Results for User %d:\n', targetUser);
  fprintf('ANOVA selected: %d features\n', length(anovaSelected));
  fprintf('MI selected: %d features\n', length(miSelected));
  fprintf('SG selected: %d features\n', length(sgSelected));
  fprintf('Combined unique features: %d\n', length(selectedFeatures{targetUser}));

  % Additional Plot: Top Features Rankings
  figure('Name', sprintf('Feature Importance - User %d', targetUser));
  
  % Calculate combined importance score
  pScores = normalize(pValues, 'range');
  mScores = normalize(miScores, 'range');
  gScores = normalize(mean(gradients, 1), 'range');
  
  % Ensure all scores have the same size
  minLength = min([length(pScores), length(mScores), length(gScores)]);
  pScores = pScores(1:minLength);
  mScores = mScores(1:minLength);
  gScores = gScores(1:minLength);
  
  combinedScores = pScores + mScores + gScores;
  [sortedScores, sortedIdx] = sort(combinedScores, 'descend');
  
  % Plot top 20 features with their individual scores
  topK = min(20, length(sortedIdx));
  subplot(2,1,1);
  bar(sortedScores(1:topK));
  title('Top 20 Features - Combined Score');
  xlabel('Rank');
  ylabel('Combined Score');
  
  % Show individual method scores for top features
  subplot(2,1,2);
  topFeatureScores = [pScores(sortedIdx(1:topK)); 
                      mScores(sortedIdx(1:topK));
                      gScores(sortedIdx(1:topK))];
  bar(normalize(topFeatureScores', 'range'));
  title('Individual Method Scores for Top Features');
  xlabel('Feature Rank');
  ylabel('Normalized Score');
  legend('ANOVA', 'MI', 'Gradient');
  
 
  % Store feature analysis results
  featureAnalysis(targetUser).pValues = pValues;
  featureAnalysis(targetUser).miScores = miScores;
  featureAnalysis(targetUser).gradientScores = mean(gradients,1);
  featureAnalysis(targetUser).selectedFeatures = selectedFeatureIdx;
  featureAnalysis(targetUser).correlationMatrix = correlationMatrix;
  featureAnalysis(targetUser).combinedScores = combinedScores;
  
  % Print feature analysis summary
  fprintf('\nFeature Analysis Summary for User %d:\n', targetUser);
  fprintf('Top 5 features by combined importance: %s\n', mat2str(sortedIdx(1:5)));
  fprintf('Average correlation between selected features: %.4f\n', ...
          mean(abs(correlationMatrix(triu(true(size(correlationMatrix)),1)))));
end

% Train one model for each user (one-vs-all approach)
models = cell(numUsers, 1);


userMetrics = zeros(numUsers, 15);  % Updated to 15 columns

% Initialize results storage
userMetrics = zeros(numUsers, 15);
userPerformance = zeros(numUsers, 3); % For timing, memory, throughput
userSimilarityData = cell(3, numUsers, numUsers);

for targetUser = userRange_min:userRange_max
  % Prepare Training set with selected features
  selectedIdx = selectedFeatures{targetUser};
  trainTargetSampleCount = size(userData(targetUser).trainFeatures, 1);
  trainImposterSampleCount = trainTargetSampleCount*(1/TrainTargetImposterRatio);
  trainSamplesPerImposter = floor(trainImposterSampleCount/(numUsers-1));

  XTrain = [userData(targetUser).trainFeatures(:, selectedIdx)];
  yTrain = ones(trainTargetSampleCount, 1);
  trainImposterFeatures = [];
  trainImposterLabels = [];
  for imposterUser = 1:numUsers
    if imposterUser ~= targetUser
      selectedIdx = randperm(size(userData(imposterUser).trainFeatures, 1), trainSamplesPerImposter);
      trainImposterFeatures = [trainImposterFeatures; userData(imposterUser).trainFeatures(selectedIdx, selectedFeatures{targetUser})];
      trainImposterLabels = [trainImposterLabels; zeros(trainSamplesPerImposter, 1)];
    end
  end

  XTrain = [XTrain; trainImposterFeatures];
  yTrain = [yTrain; trainImposterLabels];

  % Verify the train classes are balanced to the given ratio
  assert(sum(yTrain == 1) == trainTargetSampleCount);
  assert(sum(yTrain == 0) == trainImposterSampleCount);

  % Prepare Testing set with selected features
  testTargetSampleCount = size(userData(targetUser).testFeatures, 1);
  % testImposterSampleCount = trainTargetSampleCount;
  testImposterSampleCount = 324;
  testSamplesPerImposter = floor(testImposterSampleCount/(numUsers-1));

  XTest = [userData(targetUser).testFeatures(:, selectedFeatures{targetUser})];
  yTest = ones(testTargetSampleCount, 1);

  testImposterFeatures = [];
  testImposterLabels = [];
  testUserLabels = ones(testTargetSampleCount, 1) * targetUser;

  for imposterUser = 1:numUsers
    if imposterUser ~= targetUser
      selectedIdx = randperm(size(userData(imposterUser).testFeatures, 1), testSamplesPerImposter);
      testImposterFeatures = [testImposterFeatures; userData(imposterUser).testFeatures(selectedIdx, selectedFeatures{targetUser})];
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
  net.userdata.note = "Initial Feedforward Neural Network with random Leave-Out Users";
  net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
  net.userdata.dropoutRate = dropoutRate;
  net.userdata.l2RegParam = l2RegParam;
  net.userdata.performanceGoal = performanceGoal;
  net.userdata.minGrad = minGrad;
  net.userdata.earlyStoppingPatience = earlyStoppingPatience;
  net.userdata.maxEpochs = maxEpochs;
  net.userdata.learningRate = learningRate;
  net.userdata.targetUser = sprintf('User %d', targetUser);
  net.performFcn = 'crossentropy';

  % Configure layers
  net.layers{1}.transferFcn = 'logsig';
  net.layers{end}.transferFcn = 'tansig';

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
  throughput = size(XTest, 1) / inferenceTime;

  % Store raw predictions for detailed analysis
  rawPredictions = yPred;
  
  % Convert to binary predictions
  yPred = yPred > 0.5;
  yPred = double(yPred);

  % Calculate confusion matrix elements
  tp = sum(yPred == 1 & yTest == 1);  % True Positives
  tn = sum(yPred == 0 & yTest == 0);  % True Negatives
  fp = sum(yPred == 1 & yTest == 0);  % False Positives
  fn = sum(yPred == 0 & yTest == 1);  % False Negatives

  % Calculate detailed metrics
  genuinePrecision = tp / (tp + fp + eps);  % Precision for genuine attempts
  impostorPrecision = tn / (tn + fn + eps); % Precision for impostor attempts
  
  % Overall precision (weighted average)
  precision = (genuinePrecision * sum(yTest == 1) + impostorPrecision * sum(yTest == 0)) / length(yTest);
   % Overall precision (weighted average)
  Precision = (genuinePrecision * sum(yTest == 1) + impostorPrecision * sum(yTest == 0)) / length(yTest);

  fprintf('Overall Precision: %.2f%%\n', userMetrics(targetUser, 2)*100);

  
  % Calculate other metrics
  recall = tp / (tp + fn + eps);
  f1_score = 2 * (precision * recall) / (precision + recall + eps);
  specificity = tn / (tn + fp + eps);
  accuracy = (tp + tn) / (tp + tn + fp + fn);

  fpr = fp / (fp + tn + eps);
  fnr = fn / (fn + tp + eps);
  eer = (fnr + fpr) / 2;

  mcc = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps);

  % Calculate ROC and AUC
  [X,Y,T,AUC] = perfcurve(yTest, rawPredictions, 1);

  % Plot detailed confusion matrix with actual values
  figure('Name', sprintf('Detailed Confusion Matrix - User %d', targetUser));
  cm = confusionchart(yTest, yPred);
  cm.Title = sprintf('Confusion Matrix - User %d\nTP=%d, TN=%d, FP=%d, FN=%d', targetUser, tp, tn, fp, fn);
  cm.RowSummary = 'row-normalized';
  cm.ColumnSummary = 'column-normalized';

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

  fpr = fp/(fp+tn);
  fnr = fn/(fn+tp);
  eer = (fnr+fpr)/2;

  mcc = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

  [X,Y,T,AUC] = perfcurve(yTest, yPred, true);

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


  userMetrics(targetUser, :) = [accuracy, precision, recall, specificity, ...
    f1_score, mcc, fpr*100, fnr*100, eer*100, AUC, Precision , ...
    size(XTrain, 1), trainTargetSampleCount, trainImposterSampleCount, ...
    size(XTest, 1)];
  
  % Display comprehensive results for each user
  fprintf('\n==== Individual User Performance ====\n');
  fprintf('\nUser %d Results:\n', targetUser);
  fprintf('Accuracy: %.2f%%\n', userMetrics(targetUser, 1)*100);
  fprintf('Overall Precision: %.2f%%\n', userMetrics(targetUser, 2)*100);
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
  xlabel('False Positive Rate');
  ylabel('True Positive Rate');
  title(sprintf('ROC Curve - User %d (AUC = %.3f)', targetUser, AUC));
  grid on;
end

% Compute average metrics
avgMetrics = mean(userMetrics, 1);
avgPerformance = mean(userPerformance, 1);
avgOverallPrecision = mean(Precision);

% Create comprehensive results structure
results = struct(...
  'Ratio', '1:6', ...
  'AvgAccuracy', avgMetrics(1)*100, ...
  'AvgOverallPrecision', avgMetrics(11)*100, ...
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
fprintf('Hidden Layer 1: 131 neurons (tansig)\n');
fprintf('Output Layer: 1 neuron (tansig)\n');
fprintf('Training Algorithm: Scaled Conjugate Gradient (trainscg)\n');
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
  'Accuracy', 'OverallPrecision', 'Recall', 'Specificity', 'F1_Score', ...
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
  'Avg_Accuracy', 'Avg_OverallPrecision', 'Avg_Recall', 'Avg_Specificity', 'Avg_F1_Score', ...
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

    % if leaveOutUsersList(i) == j
    %   hold on;
    %   plot(j, i, 'rs', 'MarkerSize', 60);
    %   hold off;
    % end
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

% Save feature analysis results
save('feature_analysis_results.mat', 'featureAnalysis');

% Add this helper function at the end of the file
function mi = calculate_mutual_information(x, y)
    % Normalize the continuous variable x
    x = (x - min(x)) / (max(x) - min(x) + eps);
    
    % Use 10 bins for discretization
    nbins = 10;
    edges = linspace(0, 1, nbins+1);
    
    % Discretize x into bins
    [~, disc_x] = histc(x, edges);
    disc_x(disc_x == nbins+1) = nbins;
    
    % Calculate joint and marginal probabilities
    joint_hist = zeros(nbins, 2);
    for i = 1:length(x)
        if disc_x(i) > 0  % Ensure valid bin
            joint_hist(disc_x(i), y(i)+1) = joint_hist(disc_x(i), y(i)+1) + 1;
        end
    end
    
    % Convert to probabilities
    joint_p = joint_hist / (length(x) + eps);
    
    % Calculate marginal probabilities
    p_x = sum(joint_p, 2);
    p_y = sum(joint_p, 1);
    
    % Calculate mutual information
    mi = 0;
    for i = 1:nbins
        for j = 1:2
            if joint_p(i,j) > 0
                mi = mi + joint_p(i,j) * log2(joint_p(i,j) / (p_x(i) * p_y(j) + eps) + eps);
            end
        end
    end
end