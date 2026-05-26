%% PHASE 2: FEATURE EXTRACTION
% Learn: What features describe bearing health
% Build: Extract features from healthy & faulty signals

clear all; clc; close all;

fprintf('=== PHASE 2: FEATURE EXTRACTION ===\n\n');

%% Load signals from Phase 1
load('vibration_signals.mat'); % Loads: healthy, faulty, t, fs

fprintf('Loaded signals from Phase 1\n');
fprintf('Healthy signal length: %d samples\n', length(healthy));
fprintf('Faulty signal length: %d samples\n\n', length(faulty));

%% FEATURE 1: RMS (Root Mean Square)
% Meaning: Total energy in signal
% Healthy: Low energy
% Faulty: High energy

rms_healthy = sqrt(mean(healthy.^2));
rms_faulty = sqrt(mean(faulty.^2));

fprintf('FEATURE 1: RMS (Root Mean Square)\n');
fprintf('=====================================\n');
fprintf('Healthy RMS: %.4f V\n', rms_healthy);
fprintf('Faulty RMS:  %.4f V\n', rms_faulty);
fprintf('Ratio (Faulty/Healthy): %.2f x\n\n', rms_faulty/rms_healthy);
fprintf('Meaning: Faulty has %.0f%% MORE energy\n\n', (rms_faulty/rms_healthy - 1)*100);

%% FEATURE 2: PEAK (Maximum Amplitude)
% Meaning: Largest spike in signal
% Healthy: Small peaks
% Faulty: Large peaks (impacts)

peak_healthy = max(abs(healthy));
peak_faulty = max(abs(faulty));

fprintf('FEATURE 2: PEAK AMPLITUDE\n');
fprintf('=====================================\n');
fprintf('Healthy Peak: %.4f V\n', peak_healthy);
fprintf('Faulty Peak:  %.4f V\n', peak_faulty);
fprintf('Ratio: %.2f x\n\n', peak_faulty/peak_healthy);

%% FEATURE 3: CREST FACTOR
% Meaning: Peak / RMS ratio
% Healthy: Low CF (smooth waves)
% Faulty: High CF (spiky, impulsive)

cf_healthy = peak_healthy / rms_healthy;
cf_faulty = peak_faulty / rms_faulty;

fprintf('FEATURE 3: CREST FACTOR (Peak/RMS)\n');
fprintf('=====================================\n');
fprintf('Healthy CF: %.2f\n', cf_healthy);
fprintf('Faulty CF:  %.2f\n', cf_faulty);
fprintf('Meaning: Faulty has %.0f%% sharper spikes\n\n', (cf_faulty/cf_healthy - 1)*100);

%% FEATURE 4: KURTOSIS
% Meaning: Spikiness/impulsiveness of signal
% Healthy: ~3 (normal distribution)
% Faulty: >5 (very spiky, many impacts)

% Manual kurtosis calculation (no toolbox needed)
kurtosis_healthy = mean((healthy - mean(healthy)).^4) / (std(healthy).^4);
kurtosis_faulty = mean((faulty - mean(faulty)).^4) / (std(faulty).^4);

fprintf('FEATURE 4: KURTOSIS (Spikiness)\n');
fprintf('=====================================\n');
fprintf('Healthy Kurtosis: %.2f\n', kurtosis_healthy);
fprintf('Faulty Kurtosis:  %.2f\n', kurtosis_faulty);
fprintf('Meaning: Normal distribution = 3\n');
fprintf('         Faulty has %.0f%% more spikes\n\n', (kurtosis_faulty/kurtosis_healthy - 1)*100);

%% FEATURE 5: FREQUENCY DOMAIN - Peak Frequency
% Meaning: Which frequency dominates
% Healthy: Peak at 60 Hz (shaft only)
% Faulty: Peak at 240 Hz (bearing defect)

fft_healthy = abs(fft(healthy));
fft_faulty = abs(fft(faulty));
freq = (0:length(healthy)-1)*fs/length(healthy);

% Find peak frequency (skip DC component)
[~, idx_peak_healthy] = max(fft_healthy(2:500));
peak_freq_healthy = freq(idx_peak_healthy + 1);

[~, idx_peak_faulty] = max(fft_faulty(2:500));
peak_freq_faulty = freq(idx_peak_faulty + 1);

fprintf('FEATURE 5: PEAK FREQUENCY\n');
fprintf('=====================================\n');
fprintf('Healthy Peak Freq: %.1f Hz (shaft speed)\n', peak_freq_healthy);
fprintf('Faulty Peak Freq:  %.1f Hz (defect signature)\n\n', peak_freq_faulty);

%% FEATURE 6: SPECTRAL MAGNITUDE at defect frequency
% Meaning: How strong is the 240 Hz component
% Healthy: Weak or none
% Faulty: Strong (defect present)

defect_freq = 240;
defect_idx = round(defect_freq * length(healthy) / fs);

mag_at_defect_healthy = fft_healthy(defect_idx);
mag_at_defect_faulty = fft_faulty(defect_idx);

fprintf('FEATURE 6: MAGNITUDE AT 240 Hz\n');
fprintf('=====================================\n');
fprintf('Healthy at 240 Hz: %.2f\n', mag_at_defect_healthy);
fprintf('Faulty at 240 Hz:  %.2f\n', mag_at_defect_faulty);
fprintf('Ratio: %.2f x stronger in faulty\n\n', mag_at_defect_faulty/mag_at_defect_healthy);

%% FEATURE 7: Total Spectral Energy (sum of all frequencies)
% Meaning: Overall frequency domain activity
% Healthy: Concentrated energy
% Faulty: Spread out energy

energy_healthy = sum(fft_healthy(1:500).^2);
energy_faulty = sum(fft_faulty(1:500).^2);

fprintf('FEATURE 7: TOTAL SPECTRAL ENERGY\n');
fprintf('=====================================\n');
fprintf('Healthy Energy: %.2e\n', energy_healthy);
fprintf('Faulty Energy:  %.2e\n', energy_faulty);
fprintf('Ratio: %.2f x\n\n', energy_faulty/energy_healthy);

%% CREATE FEATURE TABLE
fprintf('=== FEATURE COMPARISON TABLE ===\n');
fprintf('================================================\n');
fprintf('Feature              | Healthy | Faulty  | Ratio\n');
fprintf('================================================\n');
fprintf('RMS (V)              | %.4f  | %.4f  | %.2f x\n', rms_healthy, rms_faulty, rms_faulty/rms_healthy);
fprintf('Peak (V)             | %.4f  | %.4f  | %.2f x\n', peak_healthy, peak_faulty, peak_faulty/peak_healthy);
fprintf('Crest Factor         | %.2f   | %.2f   | %.2f x\n', cf_healthy, cf_faulty, cf_faulty/cf_healthy);
fprintf('Kurtosis             | %.2f   | %.2f   | %.2f x\n', kurtosis_healthy, kurtosis_faulty, kurtosis_faulty/kurtosis_healthy);
fprintf('Peak Freq (Hz)       | %.0f   | %.0f   | -\n', peak_freq_healthy, peak_freq_faulty);
fprintf('Mag @ 240 Hz         | %.2f   | %.2f   | %.2f x\n', mag_at_defect_healthy, mag_at_defect_faulty, mag_at_defect_faulty/mag_at_defect_healthy);
fprintf('Spectral Energy      | %.2e | %.2e | %.2f x\n', energy_healthy, energy_faulty, energy_faulty/energy_healthy);
fprintf('================================================\n\n');

%% VISUALIZATION
figure('Position', [100 100 1600 900]);

% Feature comparison bar chart
features = {'RMS', 'Peak', 'Crust\nFactor', 'Kurtosis', 'Mag@240Hz', 'Spectral\nEnergy'};
healthy_vals = [rms_healthy, peak_healthy, cf_healthy, kurtosis_healthy, mag_at_defect_healthy, energy_healthy/1e4];
faulty_vals = [rms_faulty, peak_faulty, cf_faulty, kurtosis_faulty, mag_at_defect_faulty, energy_faulty/1e4];

subplot(2,3,1:2);
x = 1:length(features);
width = 0.35;
bar(x - width/2, healthy_vals, width, 'b', 'DisplayName', 'Healthy');
hold on;
bar(x + width/2, faulty_vals, width, 'r', 'DisplayName', 'Faulty');
set(gca, 'XTickLabel', features);
ylabel('Value');
title('Feature Comparison: Healthy vs Faulty');
legend;
grid on;

% Ratio plot (how different they are)
subplot(2,3,3);
ratios = [rms_faulty/rms_healthy, peak_faulty/peak_healthy, cf_faulty/cf_healthy, ...
          kurtosis_faulty/kurtosis_healthy, mag_at_defect_faulty/(mag_at_defect_healthy+0.01)];
bar(1:5, ratios, 'g');
set(gca, 'XTickLabel', {'RMS', 'Peak', 'CF', 'Kurt', 'Mag240'});
ylabel('Faulty/Healthy Ratio');
title('How Much Faulty Differs');
grid on;
hold on;
plot(1:5, ones(1,5), 'k--', 'LineWidth', 2); % Reference line at 1.0

% Time domain signals
subplot(2,3,4);
plot(t(1:1000), healthy(1:1000), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('Healthy Signal (1000 samples)');

subplot(2,3,5);
plot(t(1:1000), faulty(1:1000), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('Faulty Signal (1000 samples)');

% FFT comparison
subplot(2,3,6);
plot(freq(1:500), fft_healthy(1:500), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Healthy');
hold on;
plot(freq(1:500), fft_faulty(1:500), 'r-', 'LineWidth', 1.5, 'DisplayName', 'Faulty');
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FFT Comparison');
xlim([0 300]);
legend;

%% Save features for Phase 3
features_table = table(...
    {'Healthy'; 'Faulty'}, ...
    [rms_healthy; rms_faulty], ...
    [peak_healthy; peak_faulty], ...
    [cf_healthy; cf_faulty], ...
    [kurtosis_healthy; kurtosis_faulty], ...
    [peak_freq_healthy; peak_freq_faulty], ...
    [mag_at_defect_healthy; mag_at_defect_faulty], ...
    'VariableNames', {'Status', 'RMS', 'Peak', 'CrestFactor', 'Kurtosis', 'PeakFreq', 'Mag240Hz'});

save('extracted_features.mat', 'features_table', 'healthy_vals', 'faulty_vals');

fprintf('Features saved to: extracted_features.mat\n');
fprintf('Ready for Phase 3: Machine Learning Classification\n');