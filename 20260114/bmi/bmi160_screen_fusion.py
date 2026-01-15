import csv, time, threading, os, math
from collections import deque

import numpy as np
from ahrs.filters import Madgwick
from BMI160_i2c import Driver

OUTPUT_DIR = os.path.expanduser("~/imu_logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sensor = Driver(0x69)

ACC_RANGE_G = 2.0
GYRO_RANGE_DPS = 250.0

ACC_LSB_PER_G = (2**16) / (2*ACC_RANGE_G)
GYRO_LSB_PER_DPS = (2**16) / (2*GYRO_RANGE_DPS)

def read_imu_si():
    gx_raw, gy_raw, gz_raw, ax_raw, ay_raw, az_raw = sensor.getMotion6()
    ax_g = ax_raw / ACC_LSB_PER_G
    ay_g = ay_raw / ACC_LSB_PER_G
    az_g = az_raw / ACC_LSB_PER_G
    gx_dps = gx_raw / GYRO_LSB_PER_DPS
    gy_dps = gy_raw / GYRO_LSB_PER_DPS
    gz_dps = gz_raw / GYRO_LSB_PER_DPS
    ax = ax_g * 9.80665
    ay = ay_g * 9.80665
    az = az_g * 9.80665
    gx = math.radians(gx_dps)
    gy = math.radians(gy_dps)
    gz = math.radians(gz_dps)
    return np.array([gx, gy, gz]), np.array([ax, ay, az])

def quat_to_euler(q):
    w, x, y, z = q
    ysqr = y*y

    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0 * (x*x + ysqr)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w*y - z*x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw  # radians

# Motion classification buffers
WINDOW = 20  # ~0.2 s at 100 Hz
acc_mag_window = deque(maxlen=WINDOW)
gyro_mag_window = deque(maxlen=WINDOW)

def classify_motion(ax, ay, az, gx, gy, gz):
    acc_mag = math.sqrt(ax*ax + ay*ay + az*az)
    gyro_mag = math.sqrt(gx*gx + gy*gy + gz*gz)
    acc_mag_window.append(acc_mag)
    gyro_mag_window.append(gyro_mag)
    if len(acc_mag_window) < WINDOW:
        return "unknown"
    acc_avg = sum(acc_mag_window)/len(acc_mag_window)
    gyro_avg = sum(gyro_mag_window)/len(gyro_mag_window)
    # Tune these thresholds empirically
    if abs(acc_avg - 9.80665) < 0.1 and gyro_avg < math.radians(2):
        return "stationary"
    elif gyro_avg < math.radians(30):
        return "slow"
    else:
        return "fast"

recording = False
stop_flag = False

def record_loop(csv_path):
    global stop_flag
    madgwick = Madgwick()  # gain_imu can be tuned if needed[web:19][web:42]
    q = np.array([1.0, 0.0, 0.0, 0.0])
    last_t = time.time()

    # Smoothing memory
    r_f = p_f = y_f = 0.0
    alpha = 0.1  # smoothing factor (0..1)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t",
            "ax","ay","az",
            "gx","gy","gz",
            "qw","qx","qy","qz",
            "roll_deg","pitch_deg","yaw_deg",
            "motion_state"
        ])
        t0 = time.time()

        while not stop_flag:
            gyr, acc = read_imu_si()
            t = time.time()
            dt = t - last_t
            last_t = t

            # Madgwick fusion
            q = madgwick.updateIMU(q=q, gyr=gyr, acc=acc, dt=dt)

            # Euler conversion
            roll, pitch, yaw = quat_to_euler(q)

            # Simple low-pass smoothing on Euler
            r_f = (1-alpha)*r_f + alpha*roll
            p_f = (1-alpha)*p_f + alpha*pitch
            y_f = (1-alpha)*y_f + alpha*yaw

            # Motion classification
            ax, ay, az = acc
            gx, gy, gz = gyr
            motion_state = classify_motion(ax, ay, az, gx, gy, gz)

            t_rel = t - t0
            writer.writerow([
                t_rel,
                ax, ay, az,
                gx, gy, gz,
                q[0], q[1], q[2], q[3],
                math.degrees(r_f),
                math.degrees(p_f),
                math.degrees(y_f),
                motion_state
            ])
            f.flush()
            time.sleep(0.01)

    print(f"Recording saved to {csv_path}")

def start_recording():
    global recording, stop_flag
    if recording:
        print("Already recording.")
        return
    recording = True
    stop_flag = False
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"bmi160_fusion_{ts}.csv")
    print(f"Starting recording -> {csv_path}")
    threading.Thread(target=record_loop, args=(csv_path,), daemon=True).start()

def stop_recording():
    global recording, stop_flag
    if not recording:
        print("Not currently recording.")
        return
    print("Stopping recording...")
    stop_flag = True
    recording = False

if __name__ == "__main__":
    print("BMI160 fusion logger (screen-triggered)")
    print("Press Enter to START, Enter again to STOP, 'q'+Enter to quit.")

    while True:
        cmd = input("> ")
        if cmd.strip().lower() == "q":
            if recording:
                stop_recording()
                time.sleep(0.5)
            print("Exiting.")
            break
        elif not recording:
            start_recording()
        else:
            stop_recording()

