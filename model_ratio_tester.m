%% FeedForwardNet with Ratio Splitting Performance Benchmarking
clear all; close all; clc;
rng(100);  % For reproducibility

% Define parameters
userRange_min = 1;
userRange_max = 10;
ratios = [ ...
  1/1, ...
  1/2, ...
  1/3, ...
  1/4, ...
  1/5, ...
  1/6, ...
  1/7 ...
  ];

% Neural Network parameters
dropoutRate = 0.3;  % Dropout rate for regularization
l2RegParam = 1e-4;  % L2 regularization parameter
performanceGoal = 1e-5;  % Performance goal for training
minGrad = 1e-6;  % Minimum gradient for training
earlyStoppingPatience = 10;  % Patience for early stopping
maxEpochs = 500;  % Maximum number of training epochs
learningRate = 0.01; % Learning rate

% Define feature set pairs
featureSets = {
  {'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'},
  {'Acc_TimeD_FDay', 'Acc_TimeD_MDay'},
  {'Acc_FreqD_FDay', 'Acc_FreqD_MDay'}
  };

% Initialize storage for all results
allResults = cell(length(featureSets), 1);

% For each feature set pair
for setIdx = 1:length(featureSets)
  fprintf('\n\nProcessing feature set: %s\n', featureSets{setIdx}{1});

  % Initialize storage
  userData = struct();

  % Load data for each user
  for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);

    % Load training and test data
    trainFile = ['dataset/' userStr '_' featureSets{setIdx}{1} '.mat'];
    testFile = ['dataset/' userStr '_' featureSets{setIdx}{2} '.mat'];

    if exist(trainFile, 'file') && exist(testFile, 'file')
      trainData = load(trainFile);
      testData = load(testFile);

      userData(user).trainFeatures = trainData.(char(fieldnames(trainData)));
      userData(user).testFeatures = testData.(char(fieldnames(testData)));
    else
      fprintf('Missing data files for user %d\n', user);
    end
  end

  % Initialize results storage
  ratioResults = cell(length(ratios), 1);

  % For each ratio
  for ratioIndex = 1:length(ratios)
    currentRatio = ratios(ratioIndex);
    fprintf('\n\n--- Benchmarking Ratio 1:%d ---\n', 1/currentRatio);

    userMetrics = zeros(userRange_max, 14);

    % For each target user
    for targetUser = userRange_min:userRange_max
      % Prepare target user samples
      targetFeatures = userData(targetUser).trainFeatures;
      numTargetSamples = min(36, size(targetFeatures, 1));

      % Select target samples
      targetSamples = targetFeatures(1:numTargetSamples, :);
      targetLabels = ones(numTargetSamples, 1);

      % Calculate number of imposter samples needed based on ratio
      numImposterSamplesNeeded = round(numTargetSamples * (1/currentRatio));

      % Collect imposter samples
      imposterFeatures = [];
      imposterLabels = [];
      samplesPerImposter = ceil(numImposterSamplesNeeded / (userRange_max - 1));

      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
          imposterData = userData(imposterUser).trainFeatures;
          if ~isempty(imposterData)
            samplesToTake = min(samplesPerImposter, size(imposterData, 1));
            imposterFeatures = [imposterFeatures; imposterData(1:samplesToTake, :)];
            imposterLabels = [imposterLabels; zeros(samplesToTake, 1)];
          end
        end
      end

      % Combine and shuffle training data
      X_train = [targetSamples; imposterFeatures];
      y_train = [targetLabels; imposterLabels];

      shuffleIdx = randperm(size(X_train, 1));
      X_train = X_train(shuffleIdx, :);
      y_train = y_train(shuffleIdx, :);

      % Normalize features
      X_train = normalize(X_train, 'range');

      % Prepare test data with both positive and negative samples
      testFeaturesPos = userData(targetUser).testFeatures;
      testLabelsPos = ones(size(testFeaturesPos, 1), 1);

      % Collect negative test samples from other users
      testFeaturesNeg = [];
      testLabelsNeg = [];
      samplesPerImposter = ceil(size(testFeaturesPos, 1) / (userRange_max - 1));
      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser && ~isempty(userData(imposterUser).testFeatures)
          imposterTestData = userData(imposterUser).testFeatures;
          samplesToTake = randperm(size(imposterTestData, 1), samplesPerImposter);
          testFeaturesNeg = [testFeaturesNeg; imposterTestData(samplesToTake, :)];
          testLabelsNeg = [testLabelsNeg; zeros(samplesPerImposter, 1)];
        end
      end

      % Combine and normalize test data
      testFeatures = [testFeaturesPos; testFeaturesNeg];
      testLabels = [testLabelsPos; testLabelsNeg];
      testFeatures = normalize(testFeatures, 'range');

      % Neural Network setup and training
      net = feedforwardnet(131, 'trainscg');
      net.performFcn = 'crossentropy';

      % Configure layers
      net.layers{1}.transferFcn = 'tansig';
      net.layers{end}.transferFcn = 'tansig';

      % Configure training parameters
      net.trainParam.epochs = maxEpochs;
      net.trainParam.goal = performanceGoal;
      net.trainParam.min_grad = minGrad;
      net.performParam.regularization = l2RegParam;
      net.trainParam.max_fail = earlyStoppingPatience;
      net.trainParam.lr = learningRate;

      % Train network
      net = train(net, X_train', y_train');

      % Neural Network evaluation
      YPred = net(testFeatures')';
      YPred = YPred > 0.5;
      YPred = double(YPred);

      % Compute metrics
      testLabels = (testLabels == 1);
      predLabels = (YPred == 1);

      % Compute confusion matrix values
      tp = sum(predLabels & testLabels);
      fp = sum(predLabels & ~testLabels);
      fn = sum(~predLabels & testLabels);
      tn = sum(~predLabels & ~testLabels);

      % Calculate metrics
      accuracy = sum(YPred == testLabels)/numel(testLabels);
      precision = tp/(tp + fp);
      recall = tp/(tp + fn);
      f1_score = 2 * (precision * recall)/(precision + recall);

      % % Create and plot confusion matrix for current user
      % confMatrix = [tp fn; fp tn];
      % figure('Position', [100 100 400 300]);
      % confusionchart(confMatrix, {'Genuine', 'Impostor'}, ...
      %   'Title', sprintf('Confusion Matrix\nUser %d - Ratio 1:%d - %s', ...
      %   targetUser, 1/currentRatio, featureSets{setIdx}{1}));

      specificity = tn/(tn + fp);

      % Matthews Correlation Coefficient
      mcc = ((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));

      % AUC Score
      [X,Y,~,auc] = perfcurve(testLabels, YPred, true);

      % Calculate FAR and FRR at different thresholds
      FAR = X;
      FRR = 1 - Y;

      % Find the threshold where FAR and FRR are equal (EER)
      [~, eerIdx] = min(abs(FAR - FRR));
      EER = (FAR(eerIdx) + FRR(eerIdx)) / 2;

      % Store metrics
      userMetrics(targetUser, :) = [accuracy, precision, recall, specificity, ...
        f1_score, mcc, FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, auc, ...
        size(X_train, 1), size(targetSamples, 1), size(imposterFeatures, 1), ...
        size(testFeatures, 1)];
    end

    % Compute average metrics for this ratio
    avgMetrics = mean(userMetrics, 1);

    % Store results for this ratio
    ratioResults{ratioIndex} = struct(...
      'Ratio', sprintf('1:%d', 1/currentRatio), ...
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
      'AvgTargetSamples', avgMetrics(12), ...
      'AvgImposterSamples', avgMetrics(13), ...
      'AvgTestSetSize', avgMetrics(14) ...
      );

    % Display results for this ratio
    fprintf('\nResults for Ratio 1:%d\n', 1/currentRatio);
    disp(ratioResults{ratioIndex});
  end

  % Store results for this feature set
  allResults{setIdx} = ratioResults;

  % Create and display results table for current feature set
  fprintf('\n\nResults for Feature Set: %s and %s\n', featureSets{setIdx}{1}, featureSets{setIdx}{2});
  resultsTable = table(...
    cellfun(@(x) x.Ratio, ratioResults, 'UniformOutput', false), ...
    cellfun(@(x) x.AvgAccuracy, ratioResults), ...
    cellfun(@(x) x.AvgPrecision, ratioResults), ...
    cellfun(@(x) x.AvgRecall, ratioResults), ...
    cellfun(@(x) x.AvgSpecificity, ratioResults), ...
    cellfun(@(x) x.AvgF1Score, ratioResults), ...
    cellfun(@(x) x.AvgMCC, ratioResults), ...
    cellfun(@(x) x.AvgFAR, ratioResults), ...
    cellfun(@(x) x.AvgFRR, ratioResults), ...
    cellfun(@(x) x.AvgEER, ratioResults), ...
    cellfun(@(x) x.AvgAUC, ratioResults), ...
    cellfun(@(x) x.AvgTrainingSetSize, ratioResults), ...
    cellfun(@(x) x.AvgTargetSamples, ratioResults), ...
    cellfun(@(x) x.AvgImposterSamples, ratioResults), ...
    cellfun(@(x) x.AvgTestSetSize, ratioResults), ...
    'VariableNames', {...
    'Ratio', 'Accuracy', 'Precision', 'Recall', 'Specificity', ...
    'F1Score', 'MCC', 'FAR', 'FRR', 'EER', 'AUC', ...
    'TrainingSetSize', 'TargetSamples', 'ImposterSamples', 'TestSetSize'});

  disp(resultsTable);
end

% Convert ratio strings to numeric values for plotting
ratioNums = 1./ratios;  % Use the original ratio numbers directly

% Convert table variables to numeric arrays
accuracy_vals = table2array(resultsTable(:,'Accuracy'));
f1_vals = table2array(resultsTable(:,'F1Score'));
auc_vals = table2array(resultsTable(:,'AUC'));
eer_vals = table2array(resultsTable(:,'EER'));

% Plotting
figure('Position', [100 100 1200 800]);

% Colors for different feature sets
colors = {'b-o', 'r-s', 'g-^'};
legendLabels = {'TimeD+FreqD', 'TimeD', 'FreqD'};

% Convert ratio strings to numeric values for plotting
ratioNums = 1./ratios;

% Create subplots with multiple lines
metrics = {'Accuracy', 'MCC', 'FAR', 'FRR'};
titles = {'Accuracy vs Ratio', 'Matthews Correlation Coefficient(MCC) vs Ratio', 'False Acceptance Rate(FAR) vs Ratio', 'False Rejection Rate(FRR) vs Ratio'};
ylabels = {'Accuracy (%)', 'MCC', 'FAR (%)', 'FRR (%)'};

for i = 1:4
  subplot(3,2,i);
  hold on;

  for setIdx = 1:length(featureSets)
    % Extract values for current metric from results
    vals = cellfun(@(x) x.(sprintf('Avg%s', metrics{i})), allResults{setIdx});
    plot(ratioNums, vals, colors{setIdx}, 'LineWidth', 1.5);
  end

  title(titles{i});
  xlabel('Ratio (1:N)');
  ylabel(ylabels{i});
  grid on;
  legend(legendLabels, 'Location', 'best');
  hold off;
end

subplot('Position', [0.125, 0.1, 0.8, 0.2]);
hold on;
for setIdx = 1:length(featureSets)
  vals = cellfun(@(x) x.('AvgEER'), allResults{setIdx});
  plot(ratioNums, vals, colors{setIdx}, 'LineWidth', 1.5);
end
title('Equal Error Rate(EER) vs Ratio');
xlabel('Ratio (1:N)');
ylabel('Equal Error Rate(EER)');
grid on;
legend(legendLabels, 'Location', 'best');
hold off;

sgtitle('Performance Metrics Across Different Ratio Splits and Feature Sets');

% Save comprehensive results
save('ratio_splitting_performance_all_features.mat', 'allResults');
