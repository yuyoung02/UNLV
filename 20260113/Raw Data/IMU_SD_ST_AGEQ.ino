#include <M5Unified.h>
#include <MadgwickAHRS.h> 
#include <SD.h>

// --- CONFIGURATION ---
Madgwick filter;
const float SENSOR_RATE = 100.0f; // 100Hz Sampling
unsigned long last_update = 0;
bool is_recording = false;

// --- VARIABLES ---
// Raw holders (used for calculation but not printed)
float ax, ay, az;
float gx, gy, gz;

// Smoothed Data holders
float ax_s = 0, ay_s = 0, az_s = 0;
float gx_s = 0, gy_s = 0, gz_s = 0;

// Filter Settings
// ALPHA: 0.1 = Very Smooth (Slow), 0.9 = Responsive (Noisy)
const float ALPHA = 0.2; 

// Fusion Output
float roll, pitch, yaw;

File logFile;

void setup() {
  // 1. Optimized M5 Init
  auto cfg = M5.config();
  cfg.serial_baudrate = 115200;
  cfg.internal_spk = false; // Mute speaker
  cfg.internal_mic = false; // Disable mic
  M5.begin(cfg);
  
  // 2. Init IMU & Filter
  M5.Imu.begin();
  filter.begin(SENSOR_RATE);

  // 3. Init SD Card
  if (!SD.begin(GPIO_NUM_4, SPI, 25000000)) { 
    M5.Display.println("SD Failed!");
  } else {
    M5.Display.println("SD Ready.");
  }
  
  // 4. UI Setup
  M5.Display.setTextSize(2);
  M5.Display.setCursor(0, 40);
  M5.Display.println("Send 's' to Record");

  // 5. PRINT CSV HEADER (NO RAW COLUMNS)
  Serial.println("timestamp,ax_smooth,ay_smooth,az_smooth,gx_smooth,gy_smooth,gz_smooth,roll,pitch,yaw");
}

void loop() {
  M5.update();

  // --- UART TRIGGER ---
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == 's' || cmd == 'S') { 
      is_recording = !is_recording;
      
      if (is_recording) {
        M5.Display.fillScreen(GREEN);
        M5.Display.setCursor(10, 50);
        M5.Display.setTextColor(BLACK);
        M5.Display.print("RECORDING...");
        
        logFile = SD.open("/imu_smooth.csv", FILE_APPEND);
        if (!logFile) {
           logFile = SD.open("/imu_smooth.csv", FILE_WRITE);
           // Write Header to File
           logFile.println("timestamp,ax_smooth,ay_smooth,az_smooth,gx_smooth,gy_smooth,gz_smooth,roll,pitch,yaw"); 
        }
      } else {
        M5.Display.fillScreen(BLACK);
        M5.Display.setCursor(10, 50);
        M5.Display.setTextColor(WHITE);
        M5.Display.print("STOPPED");
        if (logFile) logFile.close();
      }
    }
  }
  
  // --- SENSOR LOOP (100Hz) ---
  if (millis() - last_update >= (1000.0 / SENSOR_RATE)) {
    last_update = millis();
    unsigned long ts = millis();

    // 1. Get Raw Data
    M5.Imu.getAccelData(&ax, &ay, &az);
    M5.Imu.getGyroData(&gx, &gy, &gz);

    // 2. Apply Low Pass Filter (Smoothing)
    // Accelerometer Smoothing
    ax_s = (ALPHA * ax) + ((1.0 - ALPHA) * ax_s);
    ay_s = (ALPHA * ay) + ((1.0 - ALPHA) * ay_s);
    az_s = (ALPHA * az) + ((1.0 - ALPHA) * az_s);

    // Gyroscope Smoothing (NEW)
    gx_s = (ALPHA * gx) + ((1.0 - ALPHA) * gx_s);
    gy_s = (ALPHA * gy) + ((1.0 - ALPHA) * gy_s);
    gz_s = (ALPHA * gz) + ((1.0 - ALPHA) * gz_s);

    // 3. Fusion (Madgwick)
    // Note: We feed RAW data to Madgwick for fastest response, 
    // but we print SMOOTH data for analysis.
    filter.updateIMU(gx, gy, gz, ax, ay, az);
    roll  = filter.getRoll();
    pitch = filter.getPitch();
    yaw   = filter.getYaw();

    // 4. STREAM SMOOTH DATA ONLY
    Serial.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                  ts, ax_s, ay_s, az_s, gx_s, gy_s, gz_s, roll, pitch, yaw);

    // 5. LOG TO SD
    if (is_recording && logFile) {
      logFile.printf("%lu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n", 
                     ts, ax_s, ay_s, az_s, gx_s, gy_s, gz_s, roll, pitch, yaw);
    }
  }
}