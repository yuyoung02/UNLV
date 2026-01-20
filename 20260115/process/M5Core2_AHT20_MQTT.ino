#include <M5Unified.h>       // Replaces M5Core2.h
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <Adafruit_AHTX0.h>

// --- USER CONFIGURATION ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "192.168.1.100"; // Replace with your RPi IP
const int mqtt_port = 1883;
const char* location = "Lab_M5_Unit1"; 
// --------------------------

WiFiClient espClient;
PubSubClient client(espClient);
Adafruit_AHTX0 aht;

void setup_wifi() {
  delay(10);
  M5.Display.print("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    M5.Display.print(".");
  }
  M5.Display.println("\nWiFi connected");
}

void reconnect() {
  while (!client.connected()) {
    M5.Display.print("Attempting MQTT connection...");
    String clientId = "M5Core2-";
    clientId += String((uint32_t)ESP.getEfuseMac(), HEX);
    
    if (client.connect(clientId.c_str())) {
      M5.Display.println("connected");
    } else {
      M5.Display.printf("failed, rc=%d try again in 5s\n", client.state());
      delay(5000);
    }
  }
}

void setup() {
  // M5Unified handles power and display initialization automatically
  M5.begin();
  M5.Display.setTextSize(2);
  M5.Display.println("Init AHT20 & MQTT...");

  // Initialize I2C for Port A (Core2 Internal I2C is different, Port A is usually 32, 33)
  // M5Unified automatically manages Wire, but we ensure connection to Port A
  Wire.begin(32, 33); 

  if (!aht.begin(&Wire)) {
    M5.Display.println("Could not find AHT? Check wiring");
    while (1) delay(10);
  }
  M5.Display.println("AHT20 Found");

  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
}

void loop() {
  M5.update(); // Update button states/power management
  
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // 1. Get Sensor Data using Unified Sensor Interface
  sensors_event_t humidity, temp;
  aht.getEvent(&humidity, &temp); // Populates the objects with fresh data

  float t_val = temp.temperature;
  float h_val = humidity.relative_humidity;

  // 2. Create JSON Payload
  StaticJsonDocument<256> doc;
  doc["loc"] = location;
  doc["temp"] = serialized(String(t_val, 2)); // Force 2 decimal places
  doc["hum"] = serialized(String(h_val, 2));

  char buffer[256];
  serializeJson(doc, buffer);

  // 3. Publish
  client.publish("lab/temphum", buffer);

  // 4. Visual Feedback
  M5.Display.fillScreen(TFT_BLACK);
  M5.Display.setCursor(10, 50);
  M5.Display.printf("Loc: %s\n", location);
  M5.Display.printf("Temp: %.2f C\n", t_val);
  M5.Display.printf("Hum:  %.2f %%", h_val);
  
  delay(5000); 
}