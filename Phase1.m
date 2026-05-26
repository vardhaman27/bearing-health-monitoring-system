%% PHASE 1: VIBRATION SIGNAL GENERATION
% Learn: How healthy vs faulty signals differ
% Build: Generate synthetic data

clear all; clc; close all;

%% Parameters
fs = 10000;           % Sampling frequency (10 kHz) - samples per second
duration = 2;         % 2 seconds of data
t = 0:1/fs:duration;  % Time array

fprintf('=== VIBRATION SIGNAL GENERATION ===\n');
fprintf('Sampling Frequency: %d Hz\n', fs);
fprintf('Duration: %d seconds\n', duration);
fprintf('Total Samples: %d\n\n', length(t));

%% SIGNAL 1: HEALTHY BEARING
fprintf('HEALTHY BEARING:\n');
fprintf('- Shaft frequency: 60 Hz\n');
fprintf('- Low amplitude\n');
fprintf('- Clean signal\n\n');

f_shaft = 60;         % Shaft rotates at 60 Hz
A_healthy = 0.5;      % Amplitude 0.5V (low)
noise_healthy = 0.05; % Small noise

healthy = A_healthy * sin(2*pi*f_shaft*t) + ...
          noise_healthy * randn(1, length(t));

%% SIGNAL 2: FAULTY BEARING (Degraded)
fprintf('FAULTY BEARING:\n');
fprintf('- Shaft frequency: 60 Hz\n');
fprintf('- Bearing defect freq: 240 Hz (4x shaft)\n');
fprintf('- High amplitude\n');
fprintf('- Chaotic + impacts\n\n');

f_defect = 240;       % Bearing defect frequency (4x shaft freq)
A_fault1 = 1.0;       % Higher amplitude
A_fault2 = 0.7;       % Defect frequency component
noise_fault = 0.3;    % High noise

% Normal shaft vibration + defect vibrations + noise
faulty = A_fault1 * sin(2*pi*f_shaft*t) + ...
         A_fault2 * sin(2*pi*f_defect*t) + ...
         noise_fault * randn(1, length(t));

% Add impulses (impacts) every 200 samples - like bearing hitting race
impulse_interval = 200;
for i = 1:impulse_interval:length(t)
    if i+50 <= length(t)
        faulty(i:i+50) = faulty(i:i+50) + 3 * exp(-0.1*(0:50));
    end
end

%% PLOT COMPARISON
figure('Position', [100 100 1400 800]);

% Plot 1: Time domain - Healthy
subplot(2,3,1);
plot(t, healthy, 'b-', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('HEALTHY BEARING - Time Domain');
axis([0 2 -2 2]);

% Plot 2: Time domain - Faulty
subplot(2,3,2);
plot(t, faulty, 'r-', 'LineWidth', 1);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('FAULTY BEARING - Time Domain');
axis([0 2 -5 5]);

% Plot 3: Zoomed healthy (first 0.1 sec)
subplot(2,3,3);
t_zoom = t(1:1000); % First 0.1 seconds
healthy_zoom = healthy(1:1000);
plot(t_zoom, healthy_zoom, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('HEALTHY - Zoomed (0.1s)');
axis([0 0.1 -1 1]);

% Plot 4: Zoomed faulty (first 0.1 sec)
subplot(2,3,4);
faulty_zoom = faulty(1:1000);
plot(t_zoom, faulty_zoom, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Amplitude (V)');
title('FAULTY - Zoomed (0.1s)');
axis([0 0.1 -4 4]);

% Plot 5: Frequency domain - Healthy (FFT)
subplot(2,3,5);
freq = (0:length(healthy)-1)*fs/length(healthy);
fft_healthy = abs(fft(healthy));
plot(freq(1:500), fft_healthy(1:500), 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('HEALTHY - Frequency Domain');
xlim([0 500]);

% Plot 6: Frequency domain - Faulty (FFT)
subplot(2,3,6);
fft_faulty = abs(fft(faulty));
plot(freq(1:500), fft_faulty(1:500), 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FAULTY - Frequency Domain');
xlim([0 500]);

%% KEY OBSERVATIONS
fprintf('KEY OBSERVATIONS:\n');
fprintf('================\n\n');

rms_healthy = rms(healthy);
rms_faulty = rms(faulty);
fprintf('RMS (overall energy):\n');
fprintf('  Healthy: %.4f V\n', rms_healthy);
fprintf('  Faulty:  %.4f V\n', rms_faulty);
fprintf('  Faulty is %.1f%% higher\n\n', (rms_faulty/rms_healthy - 1)*100);

fprintf('FREQUENCY ANALYSIS:\n');
fprintf('  Healthy: Single peak at 60 Hz (shaft speed)\n');
fprintf('  Faulty:  Peaks at 60 Hz AND 240 Hz (defect frequency)\n');
fprintf('           + Noise floor raised (less clean)\n\n');

fprintf('TIME DOMAIN:\n');
fprintf('  Healthy: Smooth, predictable waves\n');
fprintf('  Faulty:  Spiky, chaotic, random impulses\n\n');

%% Save signals for next phase
save('vibration_signals.mat', 'healthy', 'faulty', 't', 'fs');
fprintf('Signals saved to: vibration_signals.mat\n');
fprintf('Use these in Phase 2: Feature Extraction\n');