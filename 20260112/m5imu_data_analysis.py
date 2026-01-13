import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Load Data
# Skipping the first row ('--- NEW SESSION ---') and assigning column names
df = pd.read_csv('raw_imu_log.csv', skiprows=1, header=None)
df.columns = ['Timestamp', 'Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']

# Calculate Sampling Rate automatically
time_diffs = df['Timestamp'].diff().dropna()
avg_dt = time_diffs.mean() # Average time difference in ms
fs = 1000 / avg_dt         # Sampling Frequency in Hz
print(f"Estimated Sampling Frequency: {fs:.2f} Hz")

# ---------------------------------------------------------
# 1) Visualize Raw IMU Data
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Accelerometer Plot
axes[0].plot(df['Timestamp'], df['Acc_X'], label='Acc_X', alpha=0.7)
axes[0].plot(df['Timestamp'], df['Acc_Y'], label='Acc_Y', alpha=0.7)
axes[0].plot(df['Timestamp'], df['Acc_Z'], label='Acc_Z', alpha=0.7)
axes[0].set_title('Raw Accelerometer Data')
axes[0].set_ylabel('Acceleration (g)')
axes[0].legend(loc='upper right')
axes[0].grid(True)

# Gyroscope Plot
axes[1].plot(df['Timestamp'], df['Gyro_X'], label='Gyro_X', alpha=0.7)
axes[1].plot(df['Timestamp'], df['Gyro_Y'], label='Gyro_Y', alpha=0.7)
axes[1].plot(df['Timestamp'], df['Gyro_Z'], label='Gyro_Z', alpha=0.7)
axes[1].set_title('Raw Gyroscope Data')
axes[1].set_ylabel('Angular Velocity (deg/s)')
axes[1].set_xlabel('Timestamp (ms)')
axes[1].legend(loc='upper right')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('raw_imu_data.png')
plt.close()

# ---------------------------------------------------------
# 2) Apply and Visualize Smoothing (Simple Moving Average)
# ---------------------------------------------------------
# Window size of 5 samples (approx 60ms at 81Hz)
window_size_smooth = 5
df['Acc_X_Smooth'] = df['Acc_X'].rolling(window=window_size_smooth).mean()

plt.figure(figsize=(12, 4))
plt.plot(df['Timestamp'], df['Acc_X'], label='Raw Acc_X', alpha=0.4, color='gray')
plt.plot(df['Timestamp'], df['Acc_X_Smooth'], label='Smoothed Acc_X (SMA)', linewidth=2, color='blue')
plt.title(f'Smoothed Accelerometer Data (SMA Window={window_size_smooth})')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.grid(True)
plt.savefig('smoothed_data.png')
plt.close()

# ---------------------------------------------------------
# 3) Gravity Removal (High-Pass Filter)
# ---------------------------------------------------------
def butter_highpass_filter(data, cutoff, fs, order=4):
    normal_cutoff = cutoff / (0.5 * fs)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

# Cutoff at 0.5 Hz to remove static gravity component
cutoff_freq = 0.5
df['Acc_X_NoGravity'] = butter_highpass_filter(df['Acc_X'], cutoff_freq, fs)

plt.figure(figsize=(12, 4))
plt.plot(df['Timestamp'], df['Acc_X'], label='Raw Acc_X', alpha=0.4, color='gray')
plt.plot(df['Timestamp'], df['Acc_X_NoGravity'], label='Acc_X (Gravity Removed)', linewidth=1.5, color='green')
plt.title('Gravity Removal using High-Pass Filter (0.5 Hz)')
plt.xlabel('Timestamp (ms)')
plt.ylabel('Acceleration (g)')
plt.legend()
plt.grid(True)
plt.savefig('gravity_removal.png')
plt.close()

# ---------------------------------------------------------
# 4) Feature Extraction (250ms Window)
# ---------------------------------------------------------
window_ms = 250
samples_per_window = int(window_ms / avg_dt) # ~20 samples

# Calculate Signal Magnitude Vector (SMV)
df['SMV'] = np.sqrt(df['Acc_X']**2 + df['Acc_Y']**2 + df['Acc_Z']**2)

# Create a container for features
features = pd.DataFrame()
features['Timestamp'] = df['Timestamp']

# Mean and Std Dev
features['SMV_Mean'] = df['SMV'].rolling(window=samples_per_window).mean()
features['SMV_Std'] = df['SMV'].rolling(window=samples_per_window).std()

# Fall Signature Proxy: Max SMV (Impact)
features['Max_SMV'] = df['SMV'].rolling(window=samples_per_window).max()

# Zero Crossing Rate (ZCR) on gravity-removed signal
# Counts how many times the signal crosses zero in the window
features['Acc_X_ZCR'] = df['Acc_X_NoGravity'].rolling(window=samples_per_window).apply(
    lambda x: ((x[:-1] * x[1:]) < 0).sum(), raw=True
)

# Visualize Features
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Plot 1: SMV Mean & Max (Fall Signature indicators)
axes[0].plot(features['Timestamp'], features['SMV_Mean'], label='Mean SMV', color='blue')
axes[0].plot(features['Timestamp'], features['Max_SMV'], label='Max SMV (Impact)', color='red', linestyle='--')
axes[0].set_title(f'Feature: SMV Mean & Max (Window {window_ms}ms)')
axes[0].set_ylabel('g')
axes[0].legend()
axes[0].grid(True)

# Plot 2: Standard Deviation
axes[1].plot(features['Timestamp'], features['SMV_Std'], label='Std Dev of SMV', color='green')
axes[1].set_title(f'Feature: Standard Deviation of SMV (Window {window_ms}ms)')
axes[1].set_ylabel('g')
axes[1].legend()
axes[1].grid(True)

# Plot 3: Zero Crossing Rate
axes[2].plot(features['Timestamp'], features['Acc_X_ZCR'], label='ZCR (Acc X)', color='purple')
axes[2].set_title(f'Feature: Zero Crossing Rate (Window {window_ms}ms)')
axes[2].set_ylabel('Count')
axes[2].set_xlabel('Timestamp (ms)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('features.png')
plt.close()