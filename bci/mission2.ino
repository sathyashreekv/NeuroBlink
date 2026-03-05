#include <WiFi.h>
#include <WiFiClientSecure.h> // Needed for Telegram (https)
#include <HTTPClient.h>
#include <math.h>       // We need this for the "std" calculation

// =======================================================
// START: model.h content (Your ML "Brain")
// =======================================================
#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class BlinkModel {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[0] <= 2129.0) {
                            return 0;
                        }

                        else {
                            if (x[1] <= 2035.35498046875) {
                                if (x[1] <= 1978.530029296875) {
                                    return 3;
                                }

                                else {
                                    if (x[0] <= 2444.5) {
                                        return 3;
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 222.78550720214844) {
                                    return 1;
                                }

                                else {
                                    return 2;
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }

// =======================================================
// END: model.h content
// =======================================================

// --- 1. WiFi Credentials ---
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// --- 2. Telegram Bot Details ---
#define TELEGRAM_BOT_TOKEN "YOUR_BOT_TOKEN"
#define TELEGRAM_CHAT_ID "YOUR_CHAT_ID"

// --- 3. Hardware Pins ---
#define SENSOR_PIN 36
#define LED_PIN 2 // Onboard LED

// --- 4. ML Model Configuration ---
const int WINDOW_SIZE = 100; // 100 samples
const int DELAY_MS = 10;     // 10ms between samples
Eloquent::ML::Port::BlinkModel model; // Create an instance of our model
float features[3];           // Array to hold [max, mean, std]
int data[WINDOW_SIZE];       // "Bucket" to hold 1-second of data
int dataIndex = 0;           // Current position in our bucket

// --- 5. NEW COOLDOWN TIMER ---
const long ALERT_COOLDOWN = 5000; // 5000ms = 5 seconds
unsigned long lastAlertTime = 0;  // Stores when the last alert was sent

// This object is needed for HTTPS (secure connection)
WiFiClientSecure client;

// --- Forward Declaration for sendAlert ---
void sendAlert(String message);

// =======================================================
// SETUP: Connects to WiFi
// =======================================================
void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  Serial.println("BCI-ML Telegram System Initialized.");

  // For Telegram, we must use a secure client.
  client.setInsecure(); // Skips certificate validation for simplicity

  // Connect to Wi-Fi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(LED_PIN, !digitalRead(LED_PIN)); // Flash LED
  }
  Serial.println("\nWiFi connected!");
  digitalWrite(LED_PIN, LOW); // LED Off
}

// =======================================================
// LOOP: The Edge Computing ML Brain
// =======================================================
void loop() {
  // 1. Collect one sample
  data[dataIndex] = analogRead(SENSOR_PIN);
  dataIndex++;

  // 2. Check if our 1-second "bucket" is full
  if (dataIndex >= WINDOW_SIZE) {
    
    // --- 3. Feature Extraction ---
    float max_val = 0;
    for(int i=0; i<WINDOW_SIZE; i++) {
      if(data[i] > max_val) max_val = data[i];
    }
    float sum = 0;
    for(int i=0; i<WINDOW_SIZE; i++) {
      sum += data[i];
    }
    float mean_val = sum / WINDOW_SIZE;
    float std_sum = 0;
    for(int i=0; i <WINDOW_SIZE; i++) {
      std_sum += pow(data[i] - mean_val, 2);
    }
    float std_val = sqrt(std_sum / WINDOW_SIZE);
    
    features[0] = max_val;
    features[1] = mean_val;
    features[2] = std_val;

    // --- 4. Make a Prediction! ---
    int prediction = model.predict(features);

    // --- 5. Act on the Prediction (NON-BLOCKING) ---
    
    // Check if our 5-second cooldown has passed
    if (millis() - lastAlertTime > ALERT_COOLDOWN) {
      
      if (prediction == 0) {
        Serial.println("Predicted: NOISE");
      } 
      else if (prediction == 1) {
        Serial.println("Predicted: SINGLE (Ignoring)");
      }
      else if (prediction == 2) {
        Serial.println("Predicted: DOUBLE BLINK! Sending alert...");
        sendAlert("Request for FOOD/WATER received.");
        lastAlertTime = millis(); // START COOLDOWN
      }
      else if (prediction == 3) {
        Serial.println("Predicted: TRIPLE BLINK! Sending alert...");
        sendAlert("Request for WASHROOM received.");
        lastAlertTime = millis(); // START COOLDOWN
      }
      else if (prediction == 4) {
        Serial.println("Predicted: QUADRUPLE BLINK! Sending alert...");
        sendAlert("URGENT: Help required!");
        lastAlertTime = millis(); // START COOLDOWN
      }
      
    } else {
      // We are in "cooldown" mode, just print the prediction but do nothing
      // This shows the system is still working
      if (prediction > 0) { // Don't spam "noise"
        Serial.print("Predicted: ");
        Serial.print(prediction);
        Serial.println(" (In Cooldown... Ignoring Alert)");
      }
    }
    
    // Reset our "bucket" for the next 1-second chunk
    dataIndex = 0; 
  }
  
  delay(DELAY_MS);
}

// =======================================================
// FUNCTION: Sends the alert to TELEGRAM
// =======================================================
void sendAlert(String message) {
  digitalWrite(LED_PIN, HIGH);
  HTTPClient http;

  // This is the full URL for the Telegram API
  String url = "https://api.telegram.org/bot" + String(TELEGRAM_BOT_TOKEN) + 
               "/sendMessage?chat_id=" + String(TELEGRAM_CHAT_ID) + 
               "&text=" + message;

  // URL-encode the message (replaces spaces with %20)
  url.replace(" ", "%20");
               
  Serial.print("Sending alert: ");
  Serial.println(message);
  
  if (http.begin(client, url)) { // Use the secure client
    int httpCode = http.GET(); // Send the request
  
    if (httpCode > 0) {
      Serial.printf("[HTTP] Alert sent, code: %d\n", httpCode);
    } else {
      Serial.printf("[HTTP] Alert failed, error: %s\n", http.errorToString(httpCode).c_str());
    }
    http.end();
  } else {
    Serial.println("Failed to begin HTTP client for Telegram.");
  }
  
  digitalWrite(LED_PIN, LOW);
}
