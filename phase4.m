%% PHASE 4: ROBUST TRAINING WITH DIVERSE DATA
% Learn: How diverse training improves model
% Build: Better classifier trained on 100+ varied samples

clear all; clc; close all;

fprintf('=== PHASE 4: DIVERSE BEARING DATA ===\n\n');

%% Parameters
fs = 10000;
duration = 2;
t = 0:1/fs:duration;

fprintf('Generating DIVERSE training data...\n');
fprintf('Including: Different speeds, bearing types, fault severities\n\n');

%% GENERATE DIVERSE TRAINING DATA
n_healthy_diverse = 30;
n_early_fault = 20;      % Early stage fault (small defect)
n_moderate_fault = 20;   % Moderate fault (growing defect)
n_severe_fault = 30;     % Severe fault (large defect)

n_total = n_healthy_diverse + n_early_fault + n_moderate_fault + n_severe_fault;

all_features = zeros(n_total, 7);
all_labels = zeros(n_total, 1);    % 0=healthy, 1=early, 2=moderate, 3=severe
all_severity = zeros(n_total, 1);  % Actual severity level

idx = 1;

%% 1. HEALTHY BEARINGS (Diverse conditions)
fprintf('Generating %d HEALTHY samples...\n', n_healthy_diverse);

for i = 1:n_healthy_diverse
    % Variable shaft speeds (50-70 Hz)
    f_shaft = 50 + rand()*20;
    
    % Variable healthy amplitudes
    A_healthy = 0.3 + rand()*0.2;
    
    % Variable noise levels
    noise_level = 0.05 + rand()*0.05;
    
    signal = A_healthy * sin(2*pi*f_shaft*t) + noise_level * randn(1, length(t));
    
    all_features(idx, :) = extract_features(signal, fs);
    all_labels(idx) = 0;      % Label: healthy
    all_severity(idx) = 0;
    idx = idx + 1;
end

%% 2. EARLY FAULT (Small defect)
fprintf('Generating %d EARLY FAULT samples...\n', n_early_fault);

for i = 1:n_early_fault
    f_shaft = 50 + rand()*20;
    f_defect = f_shaft * (3.8 + rand()*0.4);  % 3.8-4.2x shaft
    
    % Small defect amplitude
    A_shaft = 0.6 + rand()*0.2;
    A_defect = 0.3 + rand()*0.2;  % Small defect
    
    noise_level = 0.1 + rand()*0.05;
    
    signal = A_shaft * sin(2*pi*f_shaft*t) + ...
             A_defect * sin(2*pi*f_defect*t) + ...
             noise_level * randn(1, length(t));
    
    % Small impacts
    impulse_interval = 400;
    for j = 1:impulse_interval:length(t)
        if j+20 <= length(t)
            signal(j:j+20) = signal(j:j+20) + 1 * exp(-0.1*(0:20));
        end
    end
    
    all_features(idx, :) = extract_features(signal, fs);
    all_labels(idx) = 1;      % Label: early fault
    all_severity(idx) = 1;
    idx = idx + 1;
end

%% 3. MODERATE FAULT (Growing defect)
fprintf('Generating %d MODERATE FAULT samples...\n', n_moderate_fault);

for i = 1:n_moderate_fault
    f_shaft = 50 + rand()*20;
    f_defect = f_shaft * (3.8 + rand()*0.4);
    
    % Moderate defect
    A_shaft = 0.8 + rand()*0.2;
    A_defect = 0.6 + rand()*0.2;  % Growing defect
    
    noise_level = 0.2 + rand()*0.05;
    
    signal = A_shaft * sin(2*pi*f_shaft*t) + ...
             A_defect * sin(2*pi*f_defect*t) + ...
             noise_level * randn(1, length(t));
    
    % Moderate impacts
    impulse_interval = 250;
    for j = 1:impulse_interval:length(t)
        if j+35 <= length(t)
            signal(j:j+35) = signal(j:j+35) + 2 * exp(-0.1*(0:35));
        end
    end
    
    all_features(idx, :) = extract_features(signal, fs);
    all_labels(idx) = 2;      % Label: moderate fault
    all_severity(idx) = 2;
    idx = idx + 1;
end

%% 4. SEVERE FAULT (Large defect - imminent failure)
fprintf('Generating %d SEVERE FAULT samples...\n\n', n_severe_fault);

for i = 1:n_severe_fault
    f_shaft = 50 + rand()*20;
    f_defect = f_shaft * (3.8 + rand()*0.4);
    
    % High amplitudes - bearing about to fail
    A_shaft = 1.2 + rand()*0.3;
    A_defect = 1.0 + rand()*0.3;  % Large defect
    
    noise_level = 0.4 + rand()*0.1;
    
    signal = A_shaft * sin(2*pi*f_shaft*t) + ...
             A_defect * sin(2*pi*f_defect*t) + ...
             noise_level * randn(1, length(t));
    
    % Severe impacts - rapid deterioration
    impulse_interval = 150;
    for j = 1:impulse_interval:length(t)
        if j+50 <= length(t)
            signal(j:j+50) = signal(j:j+50) + 4 * exp(-0.1*(0:50));
        end
    end
    
    all_features(idx, :) = extract_features(signal, fs);
    all_labels(idx) = 3;      % Label: severe fault
    all_severity(idx) = 3;
    idx = idx + 1;
end

%% FEATURE STATISTICS BY CONDITION
fprintf('=== FEATURE STATISTICS BY BEARING CONDITION ===\n\n');

conditions = {'Healthy', 'Early Fault', 'Moderate Fault', 'Severe Fault'};
feature_names = {'RMS', 'Peak', 'CF', 'Kurtosis', 'PeakFreq', 'Mag240Hz', 'SpecEnergy'};

fprintf('Feature        | Healthy  | Early    | Moderate | Severe\n');
fprintf('================================================================\n');

for j = 1:7
    healthy = mean(all_features(all_labels==0, j));
    early = mean(all_features(all_labels==1, j));
    moderate = mean(all_features(all_labels==2, j));
    severe = mean(all_features(all_labels==3, j));
    
    fprintf('%-14s | %8.3f | %8.3f | %8.3f | %8.3f\n', ...
        feature_names{j}, healthy, early, moderate, severe);
end

fprintf('\n');

%% TRAIN IMPROVED CLASSIFIER (Distance-based)
fprintf('Training improved 4-class classifier...\n');
fprintf('Method: Distance-Based Classification\n');
fprintf('Strategy: Find nearest class using feature distances\n\n');

% Calculate class centroids (mean features for each class)
healthy_features = all_features(all_labels==0, :);
early_features = all_features(all_labels==1, :);
moderate_features = all_features(all_labels==2, :);
severe_features = all_features(all_labels==3, :);

centroid_healthy = mean(healthy_features);
centroid_early = mean(early_features);
centroid_moderate = mean(moderate_features);
centroid_severe = mean(severe_features);

fprintf('Class Centroids Learned:\n');
fprintf('Healthy Centroid:    RMS=%.3f, Kurtosis=%.2f, Mag240Hz=%.1f\n', ...
    centroid_healthy(1), centroid_healthy(4), centroid_healthy(6));
fprintf('Early Fault:         RMS=%.3f, Kurtosis=%.2f, Mag240Hz=%.1f\n', ...
    centroid_early(1), centroid_early(4), centroid_early(6));
fprintf('Moderate Fault:      RMS=%.3f, Kurtosis=%.2f, Mag240Hz=%.1f\n', ...
    centroid_moderate(1), centroid_moderate(4), centroid_moderate(6));
fprintf('Severe Fault:        RMS=%.3f, Kurtosis=%.2f, Mag240Hz=%.1f\n\n', ...
    centroid_severe(1), centroid_severe(4), centroid_severe(6));

%% CLASSIFY ALL SAMPLES (Distance-based)
predictions = zeros(n_total, 1);

for i = 1:n_total
    % Calculate normalized Euclidean distance to each class centroid
    % Normalize features for fair comparison
    features_norm = all_features(i, :);
    
    % Scale by standard deviation (so all features contribute equally)
    std_healthy = std(healthy_features);
    std_early = std(early_features);
    std_moderate = std(moderate_features);
    std_severe = std(severe_features);
    
    % Distance to each class
    dist_healthy = sum(((features_norm - centroid_healthy) ./ (std_healthy + 0.001)).^2);
    dist_early = sum(((features_norm - centroid_early) ./ (std_early + 0.001)).^2);
    dist_moderate = sum(((features_norm - centroid_moderate) ./ (std_moderate + 0.001)).^2);
    dist_severe = sum(((features_norm - centroid_severe) ./ (std_severe + 0.001)).^2);
    
    % Find minimum distance = closest class
    [~, closest] = min([dist_healthy, dist_early, dist_moderate, dist_severe]);
    predictions(i) = closest - 1;  % Convert to 0-3 labels
end

%% CALCULATE ACCURACY
correct = sum(predictions == all_labels);
accuracy = correct / n_total * 100;

fprintf('=== CLASSIFICATION RESULTS ===\n');
fprintf('Total samples: %d\n', n_total);
fprintf('Correct: %d / %d\n', correct, n_total);
fprintf('Accuracy: %.1f%%\n\n', accuracy);

% Per-class accuracy
for c = 0:3
    class_mask = (all_labels == c);
    class_correct = sum((predictions == c) & class_mask);
    class_total = sum(class_mask);
    class_acc = class_correct / class_total * 100;
    fprintf('%s accuracy: %.1f%% (%d/%d)\n', conditions{c+1}, class_acc, class_correct, class_total);
end

fprintf('\n');

%% VISUALIZATION
figure('Position', [100 100 1600 1000]);

% Plot 1: RMS progression
subplot(2,3,1);
rms_vals = all_features(:, 1);
scatter(all_labels, rms_vals, 80, all_labels, 'filled');
colorbar;
set(gca, 'XTick', 0:3);
set(gca, 'XTickLabel', conditions);
ylabel('RMS (V)');
title('RMS: Increasing with Severity');
grid on;

% Plot 2: Kurtosis progression
subplot(2,3,2);
kurt_vals = all_features(:, 4);
scatter(all_labels, kurt_vals, 80, all_labels, 'filled');
colorbar;
set(gca, 'XTick', 0:3);
set(gca, 'XTickLabel', conditions);
ylabel('Kurtosis');
title('Kurtosis: Spikiness Increases');
grid on;

% Plot 3: Mag240Hz progression
subplot(2,3,3);
mag_vals = all_features(:, 6);
scatter(all_labels, mag_vals, 80, all_labels, 'filled');
colorbar;
set(gca, 'XTick', 0:3);
set(gca, 'XTickLabel', conditions);
ylabel('Magnitude @ 240 Hz');
title('Defect Signature: Grows with Severity');
grid on;

% Plot 4: Feature comparison
subplot(2,3,4:5);
healthy_mean = mean(all_features(all_labels==0, :));
early_mean = mean(all_features(all_labels==1, :));
moderate_mean = mean(all_features(all_labels==2, :));
severe_mean = mean(all_features(all_labels==3, :));

x = 1:7;
width = 0.2;
bar(x - 1.5*width, healthy_mean, width, 'DisplayName', 'Healthy');
hold on;
bar(x - 0.5*width, early_mean, width, 'DisplayName', 'Early');
bar(x + 0.5*width, moderate_mean, width, 'DisplayName', 'Moderate');
bar(x + 1.5*width, severe_mean, width, 'DisplayName', 'Severe');
set(gca, 'XTickLabel', feature_names);
ylabel('Feature Value');
title('All Features by Condition');
legend;
grid on;

% Plot 6: Confusion matrix for classes
subplot(2,3,6);
conf_matrix = zeros(4, 4);
for i = 0:3
    for j = 0:3
        conf_matrix(i+1, j+1) = sum((all_labels==i) & (predictions==j));
    end
end
imagesc(conf_matrix);
colorbar;
set(gca, 'XTick', 1:4);
set(gca, 'YTick', 1:4);
set(gca, 'XTickLabel', conditions);
set(gca, 'YTickLabel', conditions);
xlabel('Predicted');
ylabel('Actual');
title('Confusion Matrix');

% Add numbers
for i = 1:4
    for j = 1:4
        text(j, i, num2str(conf_matrix(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', 'white', 'FontSize', 12);
    end
end

%% PRACTICAL MAINTENANCE SCHEDULE
fprintf('=== MAINTENANCE RECOMMENDATIONS ===\n\n');
fprintf('Based on bearing condition:\n\n');
fprintf('HEALTHY:        Continue normal operation\n');
fprintf('                Next inspection: 3 months\n\n');
fprintf('EARLY FAULT:    Schedule maintenance soon\n');
fprintf('                Next inspection: 1 week\n');
fprintf('                Maintenance window: 2 weeks\n\n');
fprintf('MODERATE FAULT: Plan replacement\n');
fprintf('                Next inspection: 24 hours\n');
fprintf('                Replace within: 48 hours\n\n');
fprintf('SEVERE FAULT:   CRITICAL - Replace immediately!\n');
fprintf('                Risk: Catastrophic failure\n');
fprintf('                Action: Stop equipment, replace NOW\n\n');

%% Save improved model
save('improved_model.mat', 'centroid_healthy', 'centroid_early', 'centroid_moderate', 'centroid_severe', ...
    'all_features', 'all_labels', 'accuracy', 'n_total');

fprintf('Improved model saved: improved_model.mat\n');
fprintf('Ready for real-world deployment!\n\n');

%% HELPER FUNCTION
function features = extract_features(signal, fs)
    % Extract 7 features from vibration signal
    
    % Feature 1: RMS
    features(1) = sqrt(mean(signal.^2));
    
    % Feature 2: Peak
    features(2) = max(abs(signal));
    
    % Feature 3: Crest Factor
    features(3) = max(abs(signal)) / sqrt(mean(signal.^2));
    
    % Feature 4: Kurtosis
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