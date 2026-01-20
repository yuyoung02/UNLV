import json
import csv
import sys
from datetime import datetime
import paho.mqtt.client as mqtt

# --- CONFIGURATION ---
TOPIC = "lab/temphum"
CSV_FILE = "temphum_log.csv"
BROKER_ADDRESS = "localhost"

# Global dictionary to store running stats per location
# Structure: { "Location_Name": { "temp_sum": 0.0, "hum_sum": 0.0, "count": 0 } }
location_stats = {}

# Initialize CSV with header
try:
    with open(CSV_FILE, "r") as f:
        pass
except FileNotFoundError:
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "location", "temp", "hum", "avg_temp", "avg_hum"])
    print(f"Created new log file: {CSV_FILE}")


def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker (Result: {rc})")
    client.subscribe(TOPIC)
    print(f"Listening on topic: {TOPIC}")
    print("=" * 90)
    # Header for the table
    print(f"{'TIMESTAMP':<20} | {'LOCATION':<12} | {'TEMP (C)':<18} | {'HUMIDITY (%)':<18}")
    print(f"{'':<20} | {'':<12} | {'Current (Avg)':<18} | {'Current (Avg)':<18}")
    print("=" * 90)


def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)

        # Extract data
        loc = data.get("loc", "Unknown")
        temp = float(data.get("temp", 0.0))
        hum = float(data.get("hum", 0.0))
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- UPDATE RUNNING AVERAGES ---
        if loc not in location_stats:
            # Initialize stats for new location
            location_stats[loc] = {"temp_sum": 0.0, "hum_sum": 0.0, "count": 0}

        stats = location_stats[loc]
        stats["temp_sum"] += temp
        stats["hum_sum"] += hum
        stats["count"] += 1

        # Calculate Averages
        avg_temp = stats["temp_sum"] / stats["count"]
        avg_hum = stats["hum_sum"] / stats["count"]

        # --- DISPLAY OUTPUT ---
        # Format: "25.0 (24.8)"
        temp_str = f"{temp:.1f} ({avg_temp:.1f})"
        hum_str = f"{hum:.1f} ({avg_hum:.1f})"

        print(f"{ts:<20} | {loc:<12} | {temp_str:<18} | {hum_str:<18}")

        # --- SAVE TO CSV ---
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            # We also save the averages to the CSV now
            writer.writerow([ts, loc, temp, hum, round(avg_temp, 2), round(avg_hum, 2)])

    except json.JSONDecodeError:
        print(f"Error: Received non-JSON payload: {msg.payload}")
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