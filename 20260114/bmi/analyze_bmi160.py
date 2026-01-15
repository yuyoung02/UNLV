import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python analyzebmi160.py <your_csv_file.csv>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

print(f"Analyzed {csv_file}: {len(df)} samples, duration {df['t'].max():.1f}s")

# ----- Motion statistics -----
if 'motion_state' in df.columns:
    print("\nMotion statistics:")
    print(df['motion_state'].value_counts())
else:
    print("\nNo 'motion_state' column found (raw CSV).")
    print("Use the fusion logger (bmi160_screen_fusion.py) to get motion labels.")
    # You can still continue to plot basic signals.

# ----- Magnitudes -----
df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
df['gyro_mag'] = np.sqrt(df['gx']**2 + df['gy']**2 + df['gz']**2)

# ----- Figure -----
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# 1. Accel magnitude + gravity line
axes[0,0].plot(df['t'], df['acc_mag'], 'b-', label='Acc mag (m/s²)', linewidth=1)
axes[0,0].axhline(9.80665, color='r', ls='--', label='Gravity 9.81')
axes[0,0].set_ylabel('Acc mag')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Gyro magnitude
axes[0,1].plot(df['t'], np.degrees(df['gyro_mag']), 'g-', label='Gyro mag (deg/s)')
axes[0,1].set_ylabel('Gyro mag')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Euler angles (if present)
if {'roll_deg','pitch_deg','yaw_deg'}.issubset(df.columns):
    axes[1,0].plot(df['t'], df['roll_deg'], 'r-', label='Roll', alpha=0.7, linewidth=1)
    axes[1,0].plot(df['t'], df['pitch_deg'], 'g-', label='Pitch', alpha=0.7, linewidth=1)
    axes[1,0].plot(df['t'], df['yaw_deg'], 'b-', label='Yaw', alpha=0.7, linewidth=1)
    axes[1,0].set_ylabel('Euler (°)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
else:
    axes[1,0].text(0.5, 0.5, 'No Euler columns found', ha='center', va='center')
    axes[1,0].set_axis_off()

# 4. Quaternion components (if present)
quat_cols = {'qw','qx','qy','qz'}
if quat_cols.issubset(df.columns):
    axes[1,1].plot(df['t'], df['qw'], 'purple', label='qw', alpha=0.8)
    axes[1,1].plot(df['t'], df['qx'], 'orange', label='qx', alpha=0.8)
    axes[1,1].plot(df['t'], df['qy'], 'brown', label='qy', alpha=0.8)
    axes[1,1].plot(df['t'], df['qz'], 'gray', label='qz', alpha=0.8)
    axes[1,1].set_ylabel('Quaternion')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
else:
    axes[1,1].text(0.5, 0.5, 'No quaternion columns found', ha='center', va='center')
    axes[1,1].set_axis_off()

# 5. Motion states timeline (if motion_state present)
if 'motion_state' in df.columns:
    state_map = {'stationary': 0, 'slow': 1, 'fast': 2}
    df['state_num'] = df['motion_state'].map(state_map)
    df['state_num'] = df['state_num'].fillna(-1)  # unknown -> -1

    colors = {'stationary':'green', 'slow':'orange', 'fast':'red', 'unknown':'blue'}

    # scatter per state so colors are valid (no NaN)
    for state, num in {**state_map, 'unknown': -1}.items():
        mask = df['state_num'] == num
        if mask.any():
            axes[2,0].scatter(
                df.loc[mask, 't'],
                df.loc[mask, 'state_num'],
                c=colors[state],
                s=2,
                label=state,
                alpha=0.7
            )

    axes[2,0].plot(df['t'], np.degrees(df['gyro_mag']), 'k-', alpha=0.3, linewidth=0.5)
    axes[2,0].set_yticks([-1, 0, 1, 2])
    axes[2,0].set_yticklabels(['unknown', 'stationary', 'slow', 'fast'])
    axes[2,0].set_ylabel('Motion state')
    axes[2,0].legend()
    axes[2,0].grid(True, alpha=0.3)

    # 6. Pie chart of motion distribution
    state_counts = df['motion_state'].value_counts()
    axes[2,1].pie(
        state_counts.values,
        labels=state_counts.index,
        autopct='%1.1f%%',
        startangle=90
    )
    axes[2,1].set_title('Motion Distribution')
else:
    axes[2,0].text(0.5, 0.5, 'No motion_state column', ha='center', va='center')
    axes[2,0].set_axis_off()
    axes[2,1].set_axis_off()

plt.suptitle(f'BMI160 Analysis: {csv_file}', fontsize=16)
plt.tight_layout()
out_png = csv_file.replace('.csv', '_analysis.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSaved plot: {out_png}")

