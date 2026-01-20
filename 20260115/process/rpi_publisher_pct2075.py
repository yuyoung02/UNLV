import time
import json
import board
import adafruit_pct2075
import paho.mqtt.client as mqtt

# --- USER CONFIGURATION ---
# IP Address of your Master RPi Zero W2 running the broker
MQTT_BROKER_IP = "192.168.1.100"
MQTT_PORT = 1883
MQTT_TOPIC = "lab/temphum"

# Unique Location Identity for this specific device
LOCATION_NAME = "RPI_Node_PCT_East"

# --- HARDWARE SETUP ---
try:
    # Initialize I2C bus (SCL=GPIO3, SDA=GPIO2)
    i2c = board.I2C()

    # Initialize PCT2075 Sensor
    # Note: Default address is usually 0x37.
    # If using a generic breakout, it might be 0x48.
    sensor = adafruit_pct2075.PCT2075(i2c)

    print(f"Sensor PCT2075 initialized successfully at {LOCATION_NAME}")
except ValueError as e:
    print(f"Sensor initialization failed: {e}")
    print("Tip: Check I2C address. Try 'i2cdetect -y 1' to find it.")
    exit(1)
except Exception as e:
    print(f"Error: {e}")
    exit(1)


# --- MQTT SETUP ---
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to Master Broker at {MQTT_BROKER_IP}")
    else:
        print(f"Failed to connect, return code {rc}")


client = mqtt.Client()
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
    client.loop_start()  # Start background network loop
except Exception as e:
    print(f"Could not connect to MQTT Broker: {e}")
    exit(1)

# --- MAIN LOOP ---
try:
    while True:
        # 1. Read Temperature
        try:
            current_temp = sensor.temperature
        except OSError:
            print("Warning: Sensor read error (I/O). Retrying...")
            time.sleep(1)
            continue

        # 2. Prepare Payload
        # We send 'None' for humidity because PCT2075 cannot measure it.
        # This keeps the JSON compatible with your Master consumer script.
        payload = {
            "loc": LOCATION_NAME,
            "temp": round(current_temp, 2),
            "hum": None
        }
        json_payload = json.dumps(payload)

        # 3. Publish
        print(f"Sending: {json_payload}")
        client.publish(MQTT_TOPIC, json_payload)

        # 4. Wait
        time.sleep(5)

except KeyboardInterrupt:
    print("\nStopping...")
    client.loop_stop()
    client.disconnect()