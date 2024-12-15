%% FeedForwardNet with Different Testing Scenarios
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
dropoutRate = 0.3;  % Dropout rate for regularization
l2RegParam = 1e-4;  % L2 regularization parameter
performanceGoal = 1e-5;  % Performance goal for training
minGrad = 1e-6;  % Minimum gradient for training
earlyStoppingPatience = 10;  % Patience for early stopping
maxEpochs = 500;  % Maximum number of training epochs
learningRate = 0.01; % Learning rate

% Define feature set pairs and testing scenarios
featureSets = {
  {'Acc_TimeD_FreqD_FDay', 'Acc_TimeD_FreqD_MDay'},
  {'Acc_TimeD_FDay', 'Acc_TimeD_MDay'},
  {'Acc_FreqD_FDay', 'Acc_FreqD_MDay'}
  };

scenarioNames = {'50-50 Split FDay', 'FDay-MDay Split', 'Combined 50-50 Split'};
allResults = cell(length(featureSets), length(scenarioNames));

% For each feature set pair
for setIdx = 1:length(featureSets)
  fprintf('\n\nProcessing feature set: %s\n', featureSets{setIdx}{1});

  % For each scenario
  for scenarioIdx = 1:length(scenarioNames)
    fprintf('\nTesting Scenario: %s\n', scenarioNames{scenarioIdx});

    userMetrics = zeros(userRange_max, 14);

    % For each target user
    for targetUser = userRange_min:userRange_max
      % Load data based on scenario
      switch scenarioIdx
        case 1  % 50-50 Split FDay
          fDayFile = ['dataset/' sprintf('U%02d', targetUser) '_' featureSets{setIdx}{1} '.mat'];
          data = load(fDayFile);
          allData = data.(char(fieldnames(data)));
          splitIdx = floor(size(allData, 1)/2);
          trainFeatures = allData(1:splitIdx, :);
          testFeatures = allData(splitIdx+1:end, :);

        case 2  % FDay-MDay Split
          fDayFile = ['dataset/' sprintf('U%02d', targetUser) '_' featureSets{setIdx}{1} '.mat'];
          mDayFile = ['dataset/' sprintf('U%02d', targetUser) '_' featureSets{setIdx}{2} '.mat'];
          trainData = load(fDayFile);
          testData = load(mDayFile);
          trainFeatures = trainData.(char(fieldnames(trainData)));
          testFeatures = testData.(char(fieldnames(testData)));

        case 3  % Combined 50-50 Split
          fDayFile = ['dataset/' sprintf('U%02d', targetUser) '_' featureSets{setIdx}{1} '.mat'];
          mDayFile = ['dataset/' sprintf('U%02d', targetUser) '_' featureSets{setIdx}{2} '.mat'];
          data1 = load(fDayFile);
          data2 = load(mDayFile);
          allData = [data1.(char(fieldnames(data1))); data2.(char(fieldnames(data2)))];
          splitIdx = floor(size(allData, 1)/2);
          trainFeatures = allData(1:splitIdx, :);
          testFeatures = allData(splitIdx+1:end, :);
      end

      % Prepare training data with 1:1 ratio
      numTargetSamples = size(trainFeatures, 1);
      targetSamples = trainFeatures;
      targetLabels = ones(numTargetSamples, 1);

      % Collect equal number of imposter samples
      imposterFeatures = [];
      imposterLabels = [];
      samplesPerImposter = ceil(numTargetSamples / (userRange_max - 1));

      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
          % Load imposter data based on scenario
          switch scenarioIdx
            case 1  % 50-50 Split FDay
              impFile = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{1} '.mat'];
              if exist(impFile, 'file')
                impData = load(impFile);
                imposterData = impData.(char(fieldnames(impData)));
                imposterData = imposterData(1:splitIdx, :);  % Use first half for training
              end

            case 2  % FDay-MDay Split
              impFile = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{1} '.mat'];
              if exist(impFile, 'file')
                impData = load(impFile);
                imposterData = impData.(char(fieldnames(impData)));
              end

            case 3  % Combined 50-50 Split
              impFile1 = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{1} '.mat'];
              impFile2 = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{2} '.mat'];
              if exist(impFile1, 'file') && exist(impFile2, 'file')
                impData1 = load(impFile1);
                impData2 = load(impFile2);
                allImpData = [impData1.(char(fieldnames(impData1)));
                  impData2.(char(fieldnames(impData2)))];
                imposterData = allImpData(1:splitIdx, :);  % Use first half for training
              end
          end

          if exist('imposterData', 'var') && ~isempty(imposterData)
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
      testLabelsPos = ones(size(testFeatures, 1), 1);

      % Collect negative test samples from other users
      testFeaturesNeg = [];
      testLabelsNeg = [];
      samplesPerImposter = ceil(size(testFeatures, 1) / (userRange_max - 1));

      for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
          % Load imposter test data based on scenario
          switch scenarioIdx
            case 1  % 50-50 Split FDay
              impFile = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{1} '.mat'];
              if exist(impFile, 'file')
                impData = load(impFile);
                imposterTestData = impData.(char(fieldnames(impData)));
                imposterTestData = imposterTestData(splitIdx+1:end, :);  % Use second half for testing
              end

            case 2  % FDay-MDay Split
              impFile = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{2} '.mat'];
              if exist(impFile, 'file')
                impData = load(impFile);
                imposterTestData = impData.(char(fieldnames(impData)));
              end

            case 3  % Combined 50-50 Split
              impFile1 = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{1} '.mat'];
              impFile2 = ['dataset/' sprintf('U%02d', imposterUser) '_' featureSets{setIdx}{2} '.mat'];
              if exist(impFile1, 'file') && exist(impFile2, 'file')
                impData1 = load(impFile1);
                impData2 = load(impFile2);
                allImpData = [impData1.(char(fieldnames(impData1)));
                  impData2.(char(fieldnames(impData2)))];
                imposterTestData = allImpData(splitIdx+1:end, :);  % Use second half for testing
              end
          end

          if exist('imposterTestData', 'var') && ~isempty(imposterTestData)
            samplesToTake = min(samplesPerImposter, size(imposterTestData, 1));
            randIndices = randperm(size(imposterTestData, 1), samplesToTake);
            testFeaturesNeg = [testFeaturesNeg; imposterTestData(randIndices, :)];
            testLabelsNeg = [testLabelsNeg; zeros(samplesToTake, 1)];
          end
        end
      end

      % Combine and normalize test data
      testFeatures = [testFeatures; testFeaturesNeg];  % Note: testFeatures already contains positive samples
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

      % Prevent training window from appearing
      net.trainParam.showWindow = false;
      net.trainParam.showCommandLine = false;

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

    % Store average metrics for this scenario
    avgMetrics = mean(userMetrics, 1);
    allResults{setIdx, scenarioIdx} = struct(...
      'Scenario', scenarioNames{scenarioIdx}, ...
      'FeatureSet', featureSets{setIdx}{1}, ...
      'Accuracy', avgMetrics(1)*100, ...
      'Precision', avgMetrics(2)*100, ...
      'Recall', avgMetrics(3)*100, ...
      'Specificity', avgMetrics(4)*100, ...
      'F1Score', avgMetrics(5)*100, ...
      'MCC', avgMetrics(6), ...
      'FAR', avgMetrics(7), ...
      'FRR', avgMetrics(8), ...
      'EER', avgMetrics(9), ...
      'AUC', avgMetrics(10));
  end
end

% Plotting results as bar charts
metrics = {'FAR', 'FRR', 'EER', 'Accuracy', '', 'F1Score'};
titles = {'False Acceptance Rate', 'False Rejection Rate', 'Equal Error Rate', 'Accuracy', '', 'F1-Score',};
figure('Position', [100 100 1200 800]);

for i = 1:length(metrics)
  if isempty(metrics{i})
    continue;
  end
  subplot(2,3,i);
  data = zeros(length(featureSets), length(scenarioNames));
  for setIdx = 1:length(featureSets)
    for scenIdx = 1:length(scenarioNames)
      data(setIdx, scenIdx) = allResults{setIdx, scenIdx}.(metrics{i});
    end
  end

  b = bar(data);
  title(titles{i});
  xlabel('Feature Sets');
  ylabel([titles{i} ' (%)']);
  xticklabels({'TimeD+FreqD', 'TimeD', 'FreqD'});
  legend(scenarioNames, 'Location', 'best');
  grid on;
end

sgtitle('Performance Metrics Across Different Testing Scenarios and Feature Sets');

% After plotting results, add reorganized console output
metrics = {'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1Score', 'MCC', 'FAR', 'FRR', 'EER', 'AUC'};
featureSetNames = {'TimeD+FreqD', 'TimeD', 'FreqD'};

fprintf('\n=== Performance Metrics Comparison ===\n\n');

for setIdx = 1:length(featureSetNames)
  fprintf('\n=== %s Domain ===\n', featureSetNames{setIdx});
  fprintf('%-12s', 'Metric');
  for scenIdx = 1:length(scenarioNames)
    fprintf('%-20s', scenarioNames{scenIdx});
  end
  fprintf('\n%s\n', repmat('-', 1, 12 + 20 * length(scenarioNames)));

  for metric = metrics
    fprintf('%-12s', metric{1});
    for scenIdx = 1:length(scenarioNames)
      if strcmp(metric{1}, 'MCC') || strcmp(metric{1}, 'AUC')
        % Display these metrics without percentage
        fprintf('%-20.4f', allResults{setIdx, scenIdx}.(metric{1}));
      else
        % Display other metrics with percentage
        val = allResults{setIdx, scenIdx}.(metric{1});
        fprintf('%6.2f%%             ', val);
      end
    end
    fprintf('\n');
  end
  fprintf('\n');
end

% Save results
save('testing_scenarios_results.mat', 'allResults');
