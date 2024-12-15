%% Enhanced User Authentication System with Optimized Neural Network
% Clear workspace
clear all; close all; clc;

rng(100); % For reproducibility

% Define script params
userRange_min = 1;
userRange_max = 10;

% Configuration Parameters
TrainTargetImposterRatio = 1/5; % Optimal fixed ratio 1:5
performanceGoal = 1e-5;
minGrad = 1e-6;
earlyStoppingPatience = 10;
maxEpochs = 300; % Reduced epochs to avoid overfitting
learningRate = 0.005; % Optimized learning rate
batchSize = 32; % Mini-batch size
numUsers = userRange_max - userRange_min + 1;

% Feed forward net architecture
trainFcn = 'trainscg';
hiddenLayerSizes = [108 65];
hiddenLayerActivationFcns = {'logsig'; 'logsig'};
outputLayerActivationFcn = 'tansig';
performanceFcn = 'crossentropy';

% Data File Patterns
filePatternsTrain = 'Acc_TimeD_FreqD_FDay';
filePatternsTest = 'Acc_TimeD_FreqD_MDay';

% Initialize storage datasets
userData = struct();


fprintf('Loading data for each user...\n');

% Load Data
for user = userRange_min:userRange_max
    userStr = sprintf('U%02d', user);
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

%% Step 1: Feature Selection Enhancements using Genetic Algorithm
function [selectedFeatures, featureSelectionTime] = performGeneticFeatureSelection(userData, targetUser, userRange_min, userRange_max)
tic; % Start timing
% Combine data for feature selection
X = userData(targetUser).trainFeatures;
y = ones(size(X, 1), 1);
for imposterUser = userRange_min:userRange_max
    if imposterUser ~= targetUser
        X = [X; userData(imposterUser).trainFeatures];
        y = [y; zeros(size(userData(imposterUser).trainFeatures, 1), 1)];
    end
end

% Genetic Algorithm for Feature Selection
nFeatures = size(X, 2);
options = optimoptions('ga', 'Display', 'iter', 'PopulationSize', 50, 'MaxGenerations', 20);
[selectedFeatures, ~] = ga(@(features) featureSelectionEvalGA(features, X, y), nFeatures, [], [], [], [], ...
    zeros(1, nFeatures), ones(1, nFeatures), [], options);

selectedFeatures = find(selectedFeatures > 0.5);
featureSelectionTime = toc; % End timing
end

function score = featureSelectionEvalGA(features, X, y)
% Select features based on the binary mask
selectedX = X(:, logical(features));

% Train a model (e.g., SVM) using the selected features
model = fitcsvm(selectedX, y, 'KernelFunction', 'linear');

% Perform cross-validation
cv = crossval(model, 'KFold', 5);

% Calculate the misclassification rate
mcr = kfoldLoss(cv);

% The score to minimize (e.g., misclassification rate)
score = mcr;
end

%% Step 2: Train Neural Network for Each User with Cross-Validation and Hyperparameter Tuning
% Initialize results storage
userMetrics = zeros(numUsers, 14);
userPerformance = zeros(numUsers, 3);

fprintf('Training neural network models with cross-validation and hyperparameter tuning...\n');

% Leave-Out Users list
% leaveOutUsersList = [6, 3, 2, 5, 6, 1, 9, 7, 7, 3];

selectedFeaturesPerUser = cell(numUsers, 1);
for targetUser = userRange_min:userRange_max
    fprintf('Training model for User %d...\n', targetUser);
    fprintf('+++++ ========================================================================== +++++\n');
    fprintf('\n')

    % Perform feature selection and measure time
    [selectedFeatures, featureSelectionTime] = performGeneticFeatureSelection(userData, targetUser, userRange_min, userRange_max);
    selectedFeaturesPerUser{targetUser} = selectedFeatures;
    fprintf('Feature selection completed in %.2f seconds\n', featureSelectionTime);

    % Store feature selection time in performance metrics
    userPerformance(targetUser, :) = [featureSelectionTime, 0, 0]; % Initialize with feature selection time

    % Feature Selection
    XTrain_Target = userData(targetUser).trainFeatures(:, selectedFeatures);
    trainTargetSampleCount = size(XTrain_Target, 1);
    trainImposterSampleCount = trainTargetSampleCount * 1/TrainTargetImposterRatio;
    trainSamplesPerImposter = floor(trainImposterSampleCount/(numUsers-1));

    % Collect Imposter Features
    trainImposterFeatures = [];
    trainImposterLabels = [];
    XTrain = XTrain_Target;
    yTrain = ones(trainTargetSampleCount, 1);
    for imposterUser = 1:numUsers
        if imposterUser ~= targetUser
            selectedIdx = randperm(size(userData(imposterUser).trainFeatures, 1), trainSamplesPerImposter);
            trainImposterFeatures = [trainImposterFeatures; userData(imposterUser).trainFeatures(selectedIdx, selectedFeatures)];
            trainImposterLabels = [trainImposterLabels; zeros(trainSamplesPerImposter, 1)];
        end
    end

    % Combine Features
    XTrain = [XTrain; trainImposterFeatures];
    yTrain = [yTrain; trainImposterLabels];

    % Verify the train classes are balanced to the given ratio
    assert(sum(yTrain == 1) == trainTargetSampleCount);
    assert(sum(yTrain == 0) == trainImposterSampleCount);

    % Normalize
    [XTrain, mu, sigma] = zscore(XTrain);

    % Hyperparameter Tuning with Random Search
    bestAccuracy = 0;
    bestNet = [];
    numIterations = 30; % Number of random search iterations
    tic;
    for i = 1:numIterations
        lr = 10^(-3 + rand*2); % Random learning rate between 0.001 and 0.1
        dr = 0.2 + rand*0.2; % Random dropout rate between 0.2 and 0.4
        l2 = 10^(-4 + rand*2); % Random L2 regularization between 1e-4 and 1e-2

        % Define Neural Network
        net = feedforwardnet(hiddenLayerSizes, trainFcn);
        net.userdata.note = "Initial Feedforward Neural Network with Feature Selections";
        net.userdata.trainTargetImposterRatio = sprintf("1:%d", round(1/TrainTargetImposterRatio));
        net.userdata.dropoutRate = dr;
        net.userdata.l2RegParam = l2;
        net.userdata.performanceGoal = performanceGoal;
        net.userdata.minGrad = minGrad;
        net.userdata.earlyStoppingPatience = earlyStoppingPatience;
        net.userdata.maxEpochs = maxEpochs;
        net.userdata.learningRate = learningRate;
        net.userdata.batchSize = batchSize;
        net.userdata.noOfFeatures = size(XTrain, 2);
        net.userdata.targetUser = sprintf('User %d', targetUser);

        for layerNo = 1:length(hiddenLayerActivationFcns)
            net.layers{layerNo}.transferFcn = hiddenLayerActivationFcns{layerNo};
        end
        net.layers{end}.transferFcn = outputLayerActivationFcn; % set Output Layer
        net.performFcn = performanceFcn; % Performance function
        net.performParam.regularization = l2;

        % Prevent training window from appearing
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;

        % Train Network with Cross-Validation
        net.trainParam.epochs = maxEpochs;
        net.trainParam.goal = performanceGoal;
        net.trainParam.min_grad = minGrad;
        net.trainParam.max_fail = earlyStoppingPatience;
        net.trainParam.lr = lr; % Set learning rate
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;
        [net, tr] = train(net, XTrain', yTrain'); % Ensure XTrain' and yTrain' are transposed

        % Validate Model
        yValPred = net(XTrain(tr.valInd,:)')';
        valAccuracy = sum(yValPred > 0.5 == yTrain(tr.valInd)) / length(tr.valInd);
        if valAccuracy > bestAccuracy
            bestAccuracy = valAccuracy;
            bestNet = net;
        end
    end
    trainTime = toc;

    % Save Best Model and Normalization Parameters
    models{targetUser} = bestNet;
    normalizationParams{targetUser} = struct('mu', mu, 'sigma', sigma);

    % Measure memory usage
    modelInfo = whos('net');
    memoryUsage = modelInfo.bytes / (1024^2); % Convert to MB

    % Update performance metrics to include both feature selection and training times
    userPerformance(targetUser, :) = [featureSelectionTime + trainTime, memoryUsage, 0];
end

%% Step 3: Test Models
fprintf('Testing models...\n');
for targetUser = userRange_min:userRange_max
    fprintf('\n')
    fprintf('Testing model for User %d...\n', targetUser);
    fprintf('+++++ ========================================================================== +++++\n');
    fprintf('\n')

    % Test Features
    selectedFeatures = selectedFeaturesPerUser{targetUser};
    XTest_Target = userData(targetUser).testFeatures(:, selectedFeatures);
    testTargetSampleCount = size(userData(targetUser).testFeatures, 1);
    testImposterSampleCount = testTargetSampleCount*(numUsers-1);
    testSamplesPerImposter = floor(testImposterSampleCount/(numUsers-1));

    % Collect Imposter Features
    testImposterFeatures = [];
    testImposterLabels = [];
    XTest = XTest_Target;
    yTest = ones(testTargetSampleCount, 1);
    testUserLabels = ones(testTargetSampleCount, 1) * targetUser;
    for imposterUser = userRange_min:userRange_max
        if imposterUser ~= targetUser
            selectedIdx = randperm(size(userData(imposterUser).testFeatures, 1), testSamplesPerImposter);
            testImposterFeatures = [testImposterFeatures; userData(imposterUser).testFeatures(selectedIdx, selectedFeatures)];
            testImposterLabels = [testImposterLabels; zeros(testSamplesPerImposter, 1)];
            testUserLabels = [testUserLabels; ones(testSamplesPerImposter, 1) * imposterUser];
        end
    end

    % Combine Test Data
    XTest = [XTest; testImposterFeatures];
    yTest = [yTest; testImposterLabels];
    disp(testTargetSampleCount);
    disp(testImposterSampleCount);

    % Verify the test classes are balanced
    assert(sum(yTest == 1) == testTargetSampleCount);
    assert(sum(yTest == 0) == testImposterSampleCount);

    % Normalize using the same parameters as training
    mu = normalizationParams{targetUser}.mu;
    sigma = normalizationParams{targetUser}.sigma;
    XTest = (XTest - mu) ./ sigma;

    % Predict
    net = models{targetUser};
    tic;
    yPredProb = net(XTest')'; % Get the predicted probabilities
    inferenceTime = toc;
    throughput = size(XTest, 1) / inferenceTime; % samples/second

    % Store performance metrics
    userPerformance(targetUser, 1) = userPerformance(targetUser, 1) + inferenceTime;
    userPerformance(targetUser, 3) = throughput;
    % Store Similarities
    modelUserSimilarities = [testUserLabels, yPredProb];

    % Adjust decision threshold
    threshold = 0.5; % Example threshold, you may need to tune this value
    yPred = double(yPredProb > threshold);

    % Recalculate metrics with the new threshold
    [metrics, confusionMat, X, Y, T, AUC, EER, FAR, FRR] = calculatePerformanceMetrics(yTest, yPred, yPredProb);

    % Add additional metrics to match the placeholder size
    trainSetSize = size(XTrain, 1);
    trainTargetSamples = trainTargetSampleCount;
    trainImposterSamples = trainImposterSampleCount;
    testSetSize = size(XTest, 1);

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
    userMetrics(targetUser, :) = [metrics(1), metrics(2), metrics(3), metrics(4), ...
        metrics(5), metrics(6), metrics(7), metrics(8), metrics(9), AUC, ...
        trainSetSize, trainTargetSamples, trainImposterSamples, testSetSize];

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

    % Plot Confusion Matrix
    figure;
    confusionchart(confusionMat, {'Imposter', 'Legitimate'});
    title(sprintf('Confusion Matrix for User %d', targetUser));
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
fprintf('Input Layer: No fixed size (Depends on feature selection)\n');
for i = 1:length(hiddenLayerSizes)
    fprintf('Hidden Layer %d: %d neurons (%s)\n', i, hiddenLayerSizes(i), hiddenLayerActivationFcns{i});
end
fprintf(['Output Layer: ' net.outputs{end}.size ' neuron (' outputLayerActivationFcn ')\n']);
fprintf(['Training Algorithm: ' trainFcn '\n']);
fprintf('Performance Function: Cross-Entropy\n');
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
        % 
        % if leaveOutUsersList(i) == j
        %     hold on;
        %     plot(j, i, 'rs', 'MarkerSize', 60);
        %     hold off;
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


% Function to calculate performance metrics
function [metrics, confusionMat, X, Y, T, AUC, EER, FAR, FRR] = calculatePerformanceMetrics(yTrue, yPred, yPredProb)
% Convert predictions to binary
yPredBinary = double(yPred > 0.5);

% Confusion Matrix
confusionMat = confusionmat(yTrue, yPredBinary);

% Calculate Metrics
% True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
TP = sum((yTrue == 1) & (yPredBinary == 1));
TN = sum((yTrue == 0) & (yPredBinary == 0));
FP = sum((yTrue == 0) & (yPredBinary == 1));
FN = sum((yTrue == 1) & (yPredBinary == 0));

% Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN);

% Precision
precision = TP / (TP + FP);
if isnan(precision), precision = 0; end

% Recall (Sensitivity)
recall = TP / (TP + FN);
if isnan(recall), recall = 0; end

% Specificity
specificity = TN / (TN + FP);
if isnan(specificity), specificity = 0; end

% F1 Score
f1_score = 2 * (precision * recall) / (precision + recall);
if isnan(f1_score), f1_score = 0; end

% Matthews Correlation Coefficient (MCC)
mcc_numerator = (TP * TN) - (FP * FN);
mcc_denominator = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
if mcc_denominator == 0
    mcc = 0;
else
    mcc = mcc_numerator / mcc_denominator;
end

% ROC Curve and AUC
[X, Y, T, AUC] = perfcurve(yTrue, yPredProb, 1);

% Calculate FAR and FRR at different thresholds
FAR = X;
FRR = 1 - Y;

% Find the threshold where FAR and FRR are equal (EER)
[~, eerIdx] = min(abs(FAR - FRR));
EER = (FAR(eerIdx) + FRR(eerIdx)) / 2;

% Plot the ROC curve
figure;
plot(FAR, Y, 'b-', 'LineWidth', 2);
hold on;

% Mark the EER point on the ROC curve
plot(FAR(eerIdx), Y(eerIdx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
text(FAR(eerIdx), Y(eerIdx), sprintf('  EER = %.2f%%', EER * 100), 'VerticalAlignment', 'bottom');

% Add labels and title
xlabel('False Positive Rate (FAR)');
ylabel('True Positive Rate (1 - FRR)');
title('ROC Curve');
grid on;
hold off;

% Combine metrics into a single array
metrics = [
    accuracy, precision, recall, specificity, f1_score, mcc, ...
    FAR(eerIdx)*100, FRR(eerIdx)*100, EER*100, AUC
    ];
end

