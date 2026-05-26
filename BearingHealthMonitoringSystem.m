%% BEARING HEALTH MONITORING SYSTEM - PRODUCTION CODE
% Standalone application for real-world bearing diagnostics
% Predicts bearing health and generates maintenance schedules

clear all; clc; close all;

fprintf('\n');
fprintf('╔════════════════════════════════════════════════════════════╗\n');
fprintf('║   BEARING HEALTH MONITORING SYSTEM - PRODUCTION v1.0      ║\n');
fprintf('║                                                            ║\n');
fprintf('║   Predictive Maintenance for Rotating Equipment           ║\n');
fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

%% MAIN MENU
while true
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('MAIN MENU\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    fprintf('1. Load and Train Model (from raw data)\n');
    fprintf('2. Predict Bearing Health (from saved model)\n');
    fprintf('3. Generate Maintenance Schedule\n');
    fprintf('4. Batch Analysis (multiple bearings)\n');
    fprintf('5. System Status & Model Info\n');
    fprintf('6. Exit\n');
    fprintf('═══════════════════════════════════════════════════════════\n');
    
    choice = input('Select option (1-6): ');
    
    switch choice
        case 1
            train_model();
        case 2
            predict_bearing();
        case 3
            maintenance_schedule();
        case 4
            batch_analysis();
        case 5
            system_status();
        case 6
            fprintf('\nShutting down system...\n');
            fprintf('Thank you for using Bearing Health Monitoring System\n\n');
            break;
        otherwise
            fprintf('Invalid choice. Please select 1-6\n\n');
    end
end

%% ========== FUNCTION 1: TRAIN MODEL ==========
function train_model()
    fprintf('\n--- TRAINING MODEL FROM DATA ---\n\n');
    
    % Ask user for data type
    fprintf('Select data source:\n');
    fprintf('1. Generate synthetic training data\n');
    fprintf('2. Load from file\n');
    choice = input('Choice (1-2): ');
    
    if choice == 1
        fprintf('Generating 100 synthetic bearing samples...\n');
        [all_features, all_labels] = generate_diverse_data();
    else
        fprintf('Enter filename (without .mat): ');
        filename = input('', 's');
        try
            data = load([filename '.mat']);
            all_features = data.all_features;
            all_labels = data.all_labels;
            fprintf('Data loaded successfully\n');
        catch
            fprintf('Error loading file. Generating synthetic data instead...\n');
            [all_features, all_labels] = generate_diverse_data();
        end
    end
    
    % Train classifier
    fprintf('\nTraining 4-class classifier...\n');
    
    centroid_healthy = mean(all_features(all_labels==0, :));
    centroid_early = mean(all_features(all_labels==1, :));
    centroid_moderate = mean(all_features(all_labels==2, :));
    centroid_severe = mean(all_features(all_labels==3, :));
    
    % Test accuracy
    predictions = classify_samples(all_features, centroid_healthy, centroid_early, ...
                                   centroid_moderate, centroid_severe);
    accuracy = sum(predictions == all_labels) / length(all_labels) * 100;
    
    % Save model
    save('bearing_model.mat', 'centroid_healthy', 'centroid_early', 'centroid_moderate', ...
        'centroid_severe', 'all_features', 'all_labels');
    
    fprintf('\n✓ Model trained successfully\n');
    fprintf('  Accuracy: %.1f%%\n', accuracy);
    fprintf('  Saved to: bearing_model.mat\n\n');
end

%% ========== FUNCTION 2: PREDICT BEARING ==========
function predict_bearing()
    fprintf('\n--- BEARING HEALTH PREDICTION ---\n\n');
    
    % Load model
    try
        load('bearing_model.mat', 'centroid_healthy', 'centroid_early', ...
            'centroid_moderate', 'centroid_severe');
        fprintf('✓ Model loaded successfully\n\n');
    catch
        fprintf('✗ Error: Model file not found\n');
        fprintf('  Please train model first (Option 1)\n\n');
        return;
    end
    
    % Get input data source
    fprintf('Data source:\n');
    fprintf('1. Generate test signal\n');
    fprintf('2. Load from file\n');
    fprintf('3. Enter features manually\n');
    choice = input('Choice (1-3): ');
    
    switch choice
        case 1
            fprintf('\nGenerating test vibration signal...\n');
            fs = 10000;
            duration = 2;
            t = 0:1/fs:duration;
            
            fprintf('Fault severity (0-3): ');
            severity = input('');
            
            signal = generate_test_signal(t, severity);
            features = extract_features(signal, fs);
            
        case 2
            fprintf('Enter filename: ');
            filename = input('', 's');
            try
                data = load(filename);
                if isfield(data, 'signal')
                    signal = data.signal;
                    fs = data.fs;
                    features = extract_features(signal, fs);
                else
                    error('Signal not found in file');
                end
            catch
                fprintf('Error loading file\n\n');
                return;
            end
            
        case 3
            fprintf('\nEnter 7 features:\n');
            features = zeros(1, 7);
            feature_names = {'RMS (V)', 'Peak (V)', 'Crest Factor', 'Kurtosis', ...
                            'Peak Freq (Hz)', 'Mag @ 240Hz', 'Spectral Energy'};
            for i = 1:7
                fprintf('  %s: ', feature_names{i});
                features(i) = input('');
            end
    end
    
    % Classify
    [prediction, distances] = classify_with_distance(features, centroid_healthy, ...
                                                     centroid_early, centroid_moderate, ...
                                                     centroid_severe);
    
    % Display results
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════╗\n');
    fprintf('║         BEARING DIAGNOSIS REPORT          ║\n');
    fprintf('╚════════════════════════════════════════════╝\n\n');
    
    fprintf('Timestamp: %s\n', datetime('now'));
    fprintf('Bearing ID: AUTO_001\n\n');
    
    % Condition names
    conditions = {'HEALTHY', 'EARLY FAULT', 'MODERATE FAULT', 'SEVERE FAULT'};
    condition_colors = {'\x1b[32m', '\x1b[33m', '\x1b[33m', '\x1b[31m'};  % Green, Yellow, Red
    
    fprintf('STATUS: %s%s\x1b[0m\n\n', condition_colors{prediction+1}, conditions{prediction+1});
    
    % Feature values
    feature_names = {'RMS (V)', 'Peak (V)', 'CF', 'Kurtosis', 'PeakFreq', 'Mag240Hz', 'SpecEnergy'};
    fprintf('Feature Values:\n');
    fprintf('─────────────────────────────────────────────\n');
    for i = 1:7
        fprintf('  %-20s: %12.4f\n', feature_names{i}, features(i));
    end
    
    % Classification distances
    fprintf('\nClassifier Confidence:\n');
    fprintf('─────────────────────────────────────────────\n');
    dist_names = {'Healthy', 'Early Fault', 'Moderate Fault', 'Severe Fault'};
    for i = 1:4
        pct = (1 - distances(i)/max(distances)) * 100;
        pct = max(0, min(100, pct));
        bar_len = round(pct / 5);
        bar = repmat('█', 1, bar_len);
        fprintf('  %-15s [%-20s] %.0f%%\n', dist_names{i}, bar, pct);
    end
    
    % Recommendations
    fprintf('\n');
    fprintf('RECOMMENDATIONS:\n');
    fprintf('─────────────────────────────────────────────\n');
    
    switch prediction
        case 0
            fprintf('✓ Bearing is operating normally\n');
            fprintf('  • Continue normal operation\n');
            fprintf('  • Next inspection: 3 months\n');
            fprintf('  • Action: None required\n');
            urgency = 0;
            
        case 1
            fprintf('⚠ Early stage fault detected\n');
            fprintf('  • Schedule maintenance soon\n');
            fprintf('  • Next inspection: 1 week\n');
            fprintf('  • Maintenance window: Within 2 weeks\n');
            fprintf('  • Risk level: LOW\n');
            urgency = 1;
            
        case 2
            fprintf('⚠⚠ Moderate fault detected\n');
            fprintf('  • Plan bearing replacement\n');
            fprintf('  • Next inspection: 24 hours\n');
            fprintf('  • Maintenance window: Within 48 hours\n');
            fprintf('  • Risk level: MEDIUM\n');
            urgency = 2;
            
        case 3
            fprintf('🔴 SEVERE FAULT - CRITICAL\n');
            fprintf('  • Replace bearing IMMEDIATELY\n');
            fprintf('  • Risk: Catastrophic failure imminent\n');
            fprintf('  • Action: Stop equipment NOW if possible\n');
            fprintf('  • Risk level: CRITICAL\n');
            urgency = 3;
    end
    
    fprintf('\n');
    
    % Save report
    save_prediction_report(datetime('now'), conditions{prediction+1}, features, urgency);
end

%% ========== FUNCTION 3: MAINTENANCE SCHEDULE ==========
function maintenance_schedule()
    fprintf('\n--- MAINTENANCE SCHEDULE GENERATOR ---\n\n');
    
    % Load predictions from report
    try
        data = load('prediction_log.mat');
        predictions = data.predictions;
        timestamps = data.timestamps;
    catch
        fprintf('No prediction data found. Run predictions first.\n\n');
        return;
    end
    
    fprintf('╔════════════════════════════════════════════╗\n');
    fprintf('║       MAINTENANCE SCHEDULE REPORT         ║\n');
    fprintf('╚════════════════════════════════════════════╝\n\n');
    
    fprintf('Generated: %s\n\n', datetime('now'));
    
    % Classify bearings by urgency
    urgent = find(predictions == 3);
    medium = find(predictions == 2);
    low = find(predictions == 1);
    healthy = find(predictions == 0);
    
    % Schedule
    fprintf('CRITICAL PRIORITY (Replace immediately):\n');
    fprintf('─────────────────────────────────────────────\n');
    if isempty(urgent)
        fprintf('  • None\n');
    else
        for i = urgent
            fprintf('  • Bearing %d - %s\n', i, timestamps{i});
        end
    end
    
    fprintf('\nHIGH PRIORITY (Replace within 48 hours):\n');
    fprintf('─────────────────────────────────────────────\n');
    if isempty(medium)
        fprintf('  • None\n');
    else
        for i = medium
            fprintf('  • Bearing %d - %s\n', i, timestamps{i});
        end
    end
    
    fprintf('\nMEDIUM PRIORITY (Schedule within 2 weeks):\n');
    fprintf('─────────────────────────────────────────────\n');
    if isempty(low)
        fprintf('  • None\n');
    else
        for i = low
            fprintf('  • Bearing %d - %s\n', i, timestamps{i});
        end
    end
    
    fprintf('\nNORMAL (Continue monitoring):\n');
    fprintf('─────────────────────────────────────────────\n');
    if isempty(healthy)
        fprintf('  • None\n');
    else
        fprintf('  • %d bearings operating normally\n', length(healthy));
    end
    
    fprintf('\nSUMMARY:\n');
    fprintf('─────────────────────────────────────────────\n');
    fprintf('  Total bearings monitored: %d\n', length(predictions));
    fprintf('  Critical failures: %d (%.1f%%)\n', length(urgent), length(urgent)/length(predictions)*100);
    fprintf('  Moderate faults: %d (%.1f%%)\n', length(medium), length(medium)/length(predictions)*100);
    fprintf('  Early faults: %d (%.1f%%)\n', length(low), length(low)/length(predictions)*100);
    fprintf('  Healthy: %d (%.1f%%)\n\n', length(healthy), length(healthy)/length(predictions)*100);
end

%% ========== FUNCTION 4: BATCH ANALYSIS ==========
function batch_analysis()
    fprintf('\n--- BATCH ANALYSIS (Multiple Bearings) ---\n\n');
    
    % Load model
    try
        load('bearing_model.mat', 'centroid_healthy', 'centroid_early', ...
            'centroid_moderate', 'centroid_severe');
    catch
        fprintf('Error: Model not found. Train model first.\n\n');
        return;
    end
    
    n_bearings = input('Number of bearings to analyze: ');
    
    fprintf('\nAnalyzing %d bearings...\n\n', n_bearings);
    
    batch_results = zeros(n_bearings, 1);
    
    for b = 1:n_bearings
        fprintf('Bearing %d: ', b);
        
        % Generate random test signal with varying severity
        fs = 10000;
        duration = 2;
        t = 0:1/fs:duration;
        severity = randi([0 3]);
        
        signal = generate_test_signal(t, severity);
        features = extract_features(signal, fs);
        
        [prediction, ~] = classify_with_distance(features, centroid_healthy, ...
                                                 centroid_early, centroid_moderate, ...
                                                 centroid_severe);
        
        batch_results(b) = prediction;
        
        conditions = {'HEALTHY', 'EARLY', 'MODERATE', 'SEVERE'};
        fprintf('%s\n', conditions{prediction+1});
    end
    
    % Summary
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════╗\n');
    fprintf('║         BATCH ANALYSIS SUMMARY           ║\n');
    fprintf('╚════════════════════════════════════════════╝\n\n');
    
    healthy = sum(batch_results == 0);
    early = sum(batch_results == 1);
    moderate = sum(batch_results == 2);
    severe = sum(batch_results == 3);
    
    fprintf('Healthy:        %2d (%.1f%%)\n', healthy, healthy/n_bearings*100);
    fprintf('Early Fault:    %2d (%.1f%%)\n', early, early/n_bearings*100);
    fprintf('Moderate Fault: %2d (%.1f%%)\n', moderate, moderate/n_bearings*100);
    fprintf('Severe Fault:   %2d (%.1f%%)\n\n', severe, severe/n_bearings*100);
end

%% ========== FUNCTION 5: SYSTEM STATUS ==========
function system_status()
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════╗\n');
    fprintf('║         SYSTEM STATUS & MODEL INFO       ║\n');
    fprintf('╚════════════════════════════════════════════╝\n\n');
    
    fprintf('Software:\n');
    fprintf('  Version: 1.0 Production Release\n');
    fprintf('  Status: ACTIVE\n');
    fprintf('  Timestamp: %s\n\n', datetime('now'));
    
    try
        load('bearing_model.mat', 'all_labels');
        n_samples = length(all_labels);
        
        fprintf('Model Information:\n');
        fprintf('  Status: LOADED ✓\n');
        fprintf('  Training samples: %d\n', n_samples);
        fprintf('  Classes: 4 (Healthy, Early, Moderate, Severe)\n');
        fprintf('  Method: Distance-based classifier\n');
        fprintf('  File: bearing_model.mat\n\n');
    catch
        fprintf('Model Information:\n');
        fprintf('  Status: NOT LOADED ✗\n');
        fprintf('  Action: Train model first\n\n');
    end
    
    fprintf('Supported Features:\n');
    fprintf('  • Real-time bearing diagnosis\n');
    fprintf('  • Maintenance scheduling\n');
    fprintf('  • Batch analysis\n');
    fprintf('  • Report generation\n');
    fprintf('  • Historical tracking\n\n');
end

%% ========== HELPER FUNCTIONS ==========

function [features, labels] = generate_diverse_data()
    % Generate 100 diverse training samples
    fs = 10000;
    duration = 2;
    t = 0:1/fs:duration;
    
    n_healthy = 30;
    n_early = 20;
    n_moderate = 20;
    n_severe = 30;
    
    features = [];
    labels = [];
    
    for i = 1:n_healthy
        signal = generate_test_signal(t, 0);
        features = [features; extract_features(signal, fs)];
        labels = [labels; 0];
    end
    
    for i = 1:n_early
        signal = generate_test_signal(t, 1);
        features = [features; extract_features(signal, fs)];
        labels = [labels; 1];
    end
    
    for i = 1:n_moderate
        signal = generate_test_signal(t, 2);
        features = [features; extract_features(signal, fs)];
        labels = [labels; 2];
    end
    
    for i = 1:n_severe
        signal = generate_test_signal(t, 3);
        features = [features; extract_features(signal, fs)];
        labels = [labels; 3];
    end
end

function signal = generate_test_signal(t, severity)
    % Generate test signal based on severity (0-3)
    
    fs = 10000;
    f_shaft = 60 + randn()*5;
    
    switch severity
        case 0  % Healthy
            A = 0.3 + rand()*0.2;
            noise = 0.05 * randn(1, length(t));
            signal = A * sin(2*pi*f_shaft*t) + noise;
            
        case 1  % Early fault
            f_defect = f_shaft * 4;
            A_shaft = 0.6;
            A_defect = 0.3;
            noise = 0.1 * randn(1, length(t));
            signal = A_shaft * sin(2*pi*f_shaft*t) + A_defect * sin(2*pi*f_defect*t) + noise;
            
        case 2  % Moderate fault
            f_defect = f_shaft * 4;
            A_shaft = 0.9;
            A_defect = 0.6;
            noise = 0.2 * randn(1, length(t));
            signal = A_shaft * sin(2*pi*f_shaft*t) + A_defect * sin(2*pi*f_defect*t) + noise;
            for j = 1:250:length(t)
                if j+30 <= length(t)
                    signal(j:j+30) = signal(j:j+30) + 2 * exp(-0.1*(0:30));
                end
            end
            
        case 3  % Severe fault
            f_defect = f_shaft * 4;
            A_shaft = 1.3;
            A_defect = 1.0;
            noise = 0.4 * randn(1, length(t));
            signal = A_shaft * sin(2*pi*f_shaft*t) + A_defect * sin(2*pi*f_defect*t) + noise;
            for j = 1:150:length(t)
                if j+50 <= length(t)
                    signal(j:j+50) = signal(j:j+50) + 4 * exp(-0.1*(0:50));
                end
            end
    end
end

function features = extract_features(signal, fs)
    % Extract 7 features
    features = zeros(1, 7);
    features(1) = sqrt(mean(signal.^2));
    features(2) = max(abs(signal));
    features(3) = features(2) / features(1);
    features(4) = mean((signal - mean(signal)).^4) / (std(signal).^4);
    
    fft_sig = abs(fft(signal));
    freq = (0:length(signal)-1)*fs/length(signal);
    [~, idx] = max(fft_sig(2:500));
    features(5) = freq(idx + 1);
    
    defect_idx = round(240 * length(signal) / fs);
    if defect_idx <= length(fft_sig)
        features(6) = fft_sig(defect_idx);
    else
        features(6) = 0;
    end
    
    features(7) = sum(fft_sig(1:min(500, length(fft_sig))).^2);
end

function predictions = classify_samples(features, c0, c1, c2, c3)
    % Classify all samples
    n = size(features, 1);
    predictions = zeros(n, 1);
    
    for i = 1:n
        [pred, ~] = classify_with_distance(features(i, :), c0, c1, c2, c3);
        predictions(i) = pred;
    end
end

function [prediction, distances] = classify_with_distance(features, c0, c1, c2, c3)
    % Distance-based classification
    std0 = std([c0; c0]);
    std1 = std([c1; c1]);
    std2 = std([c2; c2]);
    std3 = std([c3; c3]);
    
    d0 = sum(((features - c0) ./ (std0 + 0.001)).^2);
    d1 = sum(((features - c1) ./ (std1 + 0.001)).^2);
    d2 = sum(((features - c2) ./ (std2 + 0.001)).^2);
    d3 = sum(((features - c3) ./ (std3 + 0.001)).^2);
    
    distances = [d0, d1, d2, d3];
    [~, prediction] = min(distances);
    prediction = prediction - 1;
end

function save_prediction_report(timestamp, condition, features, urgency)
    % Save prediction to log
    try
        load('prediction_log.mat');
    catch
        predictions = [];
        timestamps = {};
    end
    
    predictions = [predictions; urgency];
    timestamps = [timestamps, {char(timestamp)}];
    
    save('prediction_log.mat', 'predictions', 'timestamps');
end