import json
import csv
from datetime import datetime
import paho.mqtt.client as mqtt

# Configuration
TOPIC = "lab/temphum"
CSV_FILE = "temphum_log.csv"
BROKER_ADDRESS = "localhost"  # Assuming running on the Pi itself

# Initialize CSV with new 'location' column if file doesn't exist
try:
    with open(CSV_FILE, "r") as f:
        pass
except FileNotFoundError:
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "location", "temp", "hum"])
    print(f"Created new log file: {CSV_FILE}")


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(TOPIC)
    print(f"Listening on topic: {TOPIC}")
    print("-" * 65)
    print(f"{'TIMESTAMP':<22} | {'LOCATION':<15} | {'TEMP':<8} | {'HUM':<8}")
    print("-" * 65)


def on_message(client, userdata, msg):
    payload = msg.payload.decode()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        data = json.loads(payload)
        # Safely get data, defaulting to "Unknown" if location is missing
        loc = data.get("loc", "Unknown")
        temp = float(data.get("temp", 0.0))
        hum = float(data.get("hum", 0.0))

        # 1. Display on Terminal
        print(f"{ts:<22} | {loc:<15} | {temp:<8.2f} | {hum:<8.2f}")

        # 2. Save to CSV
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, loc, temp, hum])

    except json.JSONDecodeError:
        print(f"Received non-JSON message: {payload}")
    except Exception as e:
        print(f"Error processing message: {e}")


# Setup MQTT Client
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

try:
    client.connect(BROKER_ADDRESS, 1883, 60)
    client.loop_forever()
except KeyboardInterrupt:
    print("\nDisconnecting...")
    client.disconnect()
except Exception as e:
    print(f"Connection failed: {e}")