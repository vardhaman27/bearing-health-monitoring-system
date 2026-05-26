%% PHASE 3: MACHINE LEARNING CLASSIFICATION
% Learn: How to train ML model for bearing diagnosis
% Build: Classifier that predicts healthy or faulty

clear all; clc; close all;

fprintf('=== PHASE 3: MACHINE LEARNING CLASSIFICATION ===\n\n');

%% Parameters
fs = 10000;           % Sampling frequency
duration = 2;         % 2 seconds per signal
t = 0:1/fs:duration;

fprintf('Generating training data...\n\n');

%% GENERATE TRAINING DATA (Multiple samples)
% 20 healthy samples + 20 faulty samples = 40 training signals

n_healthy = 20;
n_faulty = 20;
n_total = n_healthy + n_faulty;

% Storage for features
all_features = zeros(n_total, 7);  % 7 features
all_labels = zeros(n_total, 1);    % 1 = faulty, 0 = healthy

%% Generate HEALTHY samples (with variation)
fprintf('Generating %d HEALTHY samples...\n', n_healthy);

for i = 1:n_healthy
    % Healthy: shaft frequency with small noise variations
    f_shaft = 60 + randn()*2;  % Slight variation in shaft speed
    A_healthy = 0.4 + randn()*0.1;  % Slight variation in amplitude
    noise = (0.05 + randn()*0.02) * randn(1, length(t));
    
    signal = A_healthy * sin(2*pi*f_shaft*t) + noise;
    
    % Extract 7 features
    all_features(i, :) = extract_features(signal, fs);
    all_labels(i) = 0;  % Label: 0 = healthy
end

fprintf('Generated healthy samples\n\n');

%% Generate FAULTY samples (with variation)
fprintf('Generating %d FAULTY samples...\n', n_faulty);

for i = 1:n_faulty
    % Faulty: shaft + defect frequency with variation
    f_shaft = 60 + randn()*2;
    f_defect = f_shaft * (4 + randn()*0.5);  % Defect varies around 4x
    A_shaft = 1.0 + randn()*0.2;
    A_defect = 0.7 + randn()*0.2;
    noise = (0.3 + randn()*0.1) * randn(1, length(t));
    
    signal = A_shaft * sin(2*pi*f_shaft*t) + ...
             A_defect * sin(2*pi*f_defect*t) + noise;
    
    % Add impacts (every 200 samples)
    impulse_interval = 200;
    for j = 1:impulse_interval:length(t)
        if j+50 <= length(t)
            signal(j:j+50) = signal(j:j+50) + 3 * exp(-0.1*(0:50));
        end
    end
    
    % Extract 7 features
    all_features(n_healthy + i, :) = extract_features(signal, fs);
    all_labels(n_healthy + i) = 1;  % Label: 1 = faulty
end

fprintf('Generated faulty samples\n\n');

%% Display Training Data Statistics
fprintf('=== TRAINING DATA SUMMARY ===\n');
fprintf('Total samples: %d\n', n_total);
fprintf('Healthy: %d | Faulty: %d\n\n', n_healthy, n_faulty);

fprintf('Feature statistics:\n');
fprintf('Feature          | Healthy Mean | Faulty Mean | Healthy Std | Faulty Std\n');
fprintf('==========================================================================\n');
feature_names = {'RMS', 'Peak', 'CF', 'Kurtosis', 'PeakFreq', 'Mag240Hz', 'SpecEnergy'};
for j = 1:7
    healthy_data = all_features(all_labels==0, j);
    faulty_data = all_features(all_labels==1, j);
    fprintf('%-15s | %12.4f | %11.4f | %11.4f | %10.4f\n', ...
        feature_names{j}, mean(healthy_data), mean(faulty_data), ...
        std(healthy_data), std(faulty_data));
end
fprintf('\n');

%% TRAIN CLASSIFIER (Optimized Decision Rules)
fprintf('Training classifier...\n');
fprintf('Method: Optimized Rule-Based Classifier\n');
fprintf('Why: Easy to understand, no toolbox needed\n\n');

% Optimize thresholds using training data means
healthy_features = all_features(all_labels==0, :);
faulty_features = all_features(all_labels==1, :);

% Thresholds = midpoint between healthy and faulty means
rms_threshold = (mean(healthy_features(:,1)) + mean(faulty_features(:,1))) / 2;
cf_threshold = (mean(healthy_features(:,3)) + mean(faulty_features(:,3))) / 2;
kurtosis_threshold = (mean(healthy_features(:,4)) + mean(faulty_features(:,4))) / 2;
mag240_threshold = (mean(healthy_features(:,6)) + mean(faulty_features(:,6))) / 2;

fprintf('Decision Rules Learned from Data:\n');
fprintf('IF (RMS > %.3f) OR (CF > %.3f) OR (Kurtosis > %.3f) OR (Mag240Hz > %.2f)\n', ...
    rms_threshold, cf_threshold, kurtosis_threshold, mag240_threshold);
fprintf('THEN: FAULTY\n');
fprintf('ELSE: HEALTHY\n\n');

%% TEST ON TRAINING DATA
fprintf('Testing on training data...\n\n');

predictions = zeros(n_total, 1);

for i = 1:n_total
    rms_val = all_features(i, 1);
    cf_val = all_features(i, 3);
    kurt_val = all_features(i, 4);
    mag240_val = all_features(i, 6);
    
    % Decision rule: ANY condition triggers fault detection (OR logic)
    if (rms_val > rms_threshold) || (cf_val > cf_threshold) || ...
       (kurt_val > kurtosis_threshold) || (mag240_val > mag240_threshold)
        predictions(i) = 1;  % Predict: faulty
    else
        predictions(i) = 0;  % Predict: healthy
    end
end

%% Calculate Accuracy Metrics
correct = sum(predictions == all_labels);
accuracy = correct / n_total * 100;

% Confusion matrix
true_positive = sum((predictions==1) & (all_labels==1));    % Correctly identified faulty
true_negative = sum((predictions==0) & (all_labels==0));    % Correctly identified healthy
false_positive = sum((predictions==1) & (all_labels==0));   % Incorrectly called healthy faulty
false_negative = sum((predictions==0) & (all_labels==1));   % Missed faulty (dangerous!)

sensitivity = true_positive / (true_positive + false_negative) * 100;  % Detect all faults?
specificity = true_negative / (true_negative + false_positive) * 100;  % Avoid false alarms?

fprintf('=== CLASSIFICATION RESULTS ===\n');
fprintf('Correct predictions: %d / %d\n', correct, n_total);
fprintf('Accuracy: %.1f%%\n\n', accuracy);

fprintf('Confusion Matrix:\n');
fprintf('                 Predicted Healthy | Predicted Faulty\n');
fprintf('Actually Healthy |       %d         |      %d\n', true_negative, false_positive);
fprintf('Actually Faulty  |       %d         |      %d\n\n', false_negative, true_positive);

fprintf('Sensitivity (detect faults): %.1f%%\n', sensitivity);
fprintf('Specificity (avoid false alarms): %.1f%%\n\n', specificity);

%% VISUALIZATION
figure('Position', [100 100 1600 900]);

% Plot 1: Scatter plot RMS vs CF
subplot(2,3,1);
healthy_idx = find(all_labels == 0);
faulty_idx = find(all_labels == 1);
scatter(all_features(healthy_idx, 1), all_features(healthy_idx, 3), 80, 'b', 'o', 'DisplayName', 'Healthy');
hold on;
scatter(all_features(faulty_idx, 1), all_features(faulty_idx, 3), 80, 'r', 's', 'DisplayName', 'Faulty');
xlabel('RMS (V)');
ylabel('Crest Factor');
title('RMS vs Crest Factor');
legend;
grid on;
% Decision boundary
plot([rms_threshold rms_threshold], [0 5], 'k--', 'LineWidth', 2);
text(rms_threshold*1.05, 4.5, 'RMS threshold', 'FontSize', 10);

% Plot 2: RMS vs Kurtosis
subplot(2,3,2);
scatter(all_features(healthy_idx, 1), all_features(healthy_idx, 4), 80, 'b', 'o', 'DisplayName', 'Healthy');
hold on;
scatter(all_features(faulty_idx, 1), all_features(faulty_idx, 4), 80, 'r', 's', 'DisplayName', 'Faulty');
xlabel('RMS (V)');
ylabel('Kurtosis');
title('RMS vs Kurtosis');
legend;
grid on;

% Plot 3: CF vs Kurtosis
subplot(2,3,3);
scatter(all_features(healthy_idx, 3), all_features(healthy_idx, 4), 80, 'b', 'o', 'DisplayName', 'Healthy');
hold on;
scatter(all_features(faulty_idx, 3), all_features(faulty_idx, 4), 80, 'r', 's', 'DisplayName', 'Faulty');
xlabel('Crest Factor');
ylabel('Kurtosis');
title('CF vs Kurtosis');
legend;
grid on;

% Plot 4: All features comparison
subplot(2,3,4:5);
healthy_mean = mean(all_features(healthy_idx, :));
faulty_mean = mean(all_features(faulty_idx, :));
x = 1:7;
width = 0.35;
bar(x - width/2, healthy_mean, width, 'b', 'DisplayName', 'Healthy');
hold on;
bar(x + width/2, faulty_mean, width, 'r', 'DisplayName', 'Faulty');
set(gca, 'XTickLabel', feature_names);
ylabel('Feature Value');
title('Average Features: Healthy vs Faulty');
legend;
grid on;

% Plot 5: Confusion Matrix Heatmap
subplot(2,3,6);
confusion_matrix = [true_negative, false_positive; false_negative, true_positive];
imagesc(confusion_matrix);
colorbar;
set(gca, 'XTickLabel', {'Pred: Healthy', 'Pred: Faulty'});
set(gca, 'YTickLabel', {'Actual: Healthy', 'Actual: Faulty'});
title('Confusion Matrix');
text(1, 1, num2str(true_negative), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 14);
text(2, 1, num2str(false_positive), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 14);
text(1, 2, num2str(false_negative), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 14);
text(2, 2, num2str(true_positive), 'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 14);

%% EXAMPLE: PREDICT NEW SIGNAL
fprintf('=== PREDICTING NEW SIGNAL ===\n\n');

% Generate new test signal (unknown to model)
fprintf('Creating new TEST signal...\n');
f_test = 60;
A_test = 0.8;
noise_test = 0.2 * randn(1, length(t));
test_signal = A_test * sin(2*pi*f_test*t) + noise_test;

% Add small defect
impulse_interval = 300;
for j = 1:impulse_interval:length(t)
    if j+30 <= length(t)
        test_signal(j:j+30) = test_signal(j:j+30) + 2 * exp(-0.1*(0:30));
    end
end

test_features = extract_features(test_signal, fs);

% Predict
rms_test = test_features(1);
cf_test = test_features(3);
kurt_test = test_features(4);
mag240_test = test_features(6);

if (rms_test > rms_threshold) || (cf_test > cf_threshold) || ...
   (kurt_test > kurtosis_threshold) || (mag240_test > mag240_threshold)
    prediction = 1;
    pred_str = 'FAULTY';
else
    prediction = 0;
    pred_str = 'HEALTHY';
end

fprintf('\nTest Signal Features:\n');
for j = 1:7
    fprintf('%s: %.4f\n', feature_names{j}, test_features(j));
end

fprintf('\nDecision:\n');
fprintf('RMS (%.4f) > %.3f? %s\n', rms_test, rms_threshold, yesno(rms_test > rms_threshold));
fprintf('CF (%.4f) > %.3f? %s\n', cf_test, cf_threshold, yesno(cf_test > cf_threshold));
fprintf('Kurtosis (%.4f) > %.3f? %s\n', kurt_test, kurtosis_threshold, yesno(kurt_test > kurtosis_threshold));
fprintf('Mag240Hz (%.4f) > %.2f? %s\n', mag240_test, mag240_threshold, yesno(mag240_test > mag240_threshold));

fprintf('\n*** PREDICTION: %s ***\n\n', pred_str);

%% Save model
save('trained_model.mat', 'rms_threshold', 'cf_threshold', 'kurtosis_threshold', 'mag240_threshold', ...
    'all_features', 'all_labels', 'accuracy', 'sensitivity', 'specificity');

fprintf('Model saved to: trained_model.mat\n');
fprintf('Ready to deploy on real bearing data!\n');

%% HELPER FUNCTIONS
function features = extract_features(signal, fs)
    % Extract 7 features from vibration signal
    
    % Feature 1: RMS
    features(1) = sqrt(mean(signal.^2));
    
    % Feature 2: Peak
    features(2) = max(abs(signal));
    
    % Feature 3: Crest Factor
    features(3) = max(abs(signal)) / sqrt(mean(signal.^2));
    
    % Feature 4: Kurtosis (manual)
    features(4) = mean((signal - mean(signal)).^4) / (std(signal).^4);
    
    % Feature 5: Peak Frequency
    fft_sig = abs(fft(signal));
    freq = (0:length(signal)-1)*fs/length(signal);
    [~, idx] = max(fft_sig(2:500));
    features(5) = freq(idx + 1);
    
    % Feature 6: Magnitude at 240 Hz
    defect_idx = round(240 * length(signal) / fs);
    if defect_idx <= length(fft_sig)
        features(6) = fft_sig(defect_idx);
    else
        features(6) = 0;
    end
    
    % Feature 7: Spectral Energy
    features(7) = sum(fft_sig(1:min(500, length(fft_sig))).^2);
end

function result = yesno(condition)
    if condition
        result = 'YES';
    else
        result = 'NO';
    end
end