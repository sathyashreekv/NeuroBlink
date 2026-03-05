# BCI Eye Blink Detection System

A Brain-Computer Interface (BCI) system that uses machine learning to detect eye blink patterns and send alerts via Telegram. Built for ESP32 microcontrollers with edge computing capabilities.

## Overview

This project enables paralyzed or mobility-impaired individuals to communicate basic needs through eye blinks:
- **Double Blink**: Request for food/water
- **Triple Blink**: Request for washroom assistance
- **Quadruple Blink**: Urgent help required

The system uses a trained Decision Tree classifier that runs directly on the ESP32, analyzing sensor data in real-time and sending alerts to caregivers via Telegram.

## Project Structure

```
bci_data/
├── bci/
│   └── mission2.ino          # ESP32 firmware with ML model
├── noise.csv                 # Training data: baseline noise
├── single.csv                # Training data: single blink
├── double.csv                # Training data: double blink
├── triple.csv                # Training data: triple blink
├── quadrple.csv              # Training data: quadruple blink
├── train.py                  # ML training script (basic features)
└── train2.py                 # ML training script (peak counting)
```

## Hardware Requirements

- ESP32 development board
- Analog sensor (connected to GPIO 36)
- WiFi network access
- USB cable for programming

## Software Requirements

### For Training (Python)
```bash
pip install numpy scikit-learn micromlgen
```

### For Deployment (Arduino IDE)
- ESP32 board support
- WiFi library
- HTTPClient library

## Setup Instructions

### 1. Data Collection
Collect sensor readings for each blink pattern and save as CSV files:
- `noise.csv` - Background noise/no blink
- `single.csv` - Single blink patterns
- `double.csv` - Double blink patterns
- `triple.csv` - Triple blink patterns
- `quadrple.csv` - Quadruple blink patterns

CSV format: `timestamp -> value` (e.g., `22:46:16.025 -> 1938`)

### 2. Train the Model

Run either training script:

```bash
# Basic feature extraction (max, mean, std)
python train.py

# Advanced feature extraction (std, peak count)
python train2.py
```

This generates `model.h` containing the C++ Decision Tree classifier.

### 3. Configure ESP32 Firmware

Edit `bci/mission2.ino`:

```cpp
// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Telegram bot details
#define TELEGRAM_BOT_TOKEN "YOUR_BOT_TOKEN"
#define TELEGRAM_CHAT_ID "YOUR_CHAT_ID"
```

### 4. Upload to ESP32

1. Open `mission2.ino` in Arduino IDE
2. Select ESP32 board and port
3. Upload the sketch

## How It Works

### Edge Computing Pipeline

1. **Data Collection**: Samples sensor at 100Hz (10ms intervals)
2. **Windowing**: Collects 100 samples (1-second window)
3. **Feature Extraction**: Calculates max, mean, and standard deviation
4. **Prediction**: Runs Decision Tree classifier on-device
5. **Action**: Sends Telegram alert based on prediction

### ML Model

- **Algorithm**: Decision Tree (max depth: 5)
- **Features**: 
  - Maximum value in window
  - Mean value
  - Standard deviation
- **Classes**: 
  - 0: Noise
  - 1: Single blink (ignored)
  - 2: Double blink (food/water)
  - 3: Triple blink (washroom)
  - 4: Quadruple blink (urgent help)

### Alert Cooldown

5-second cooldown prevents alert spam while maintaining system responsiveness.

## Configuration Parameters

```cpp
WINDOW_SIZE = 100      // Samples per prediction window
DELAY_MS = 10          // Sampling interval (ms)
ALERT_COOLDOWN = 5000  // Cooldown between alerts (ms)
SENSOR_PIN = 36        // Analog input pin
LED_PIN = 2            // Status LED pin
```

## Training Scripts

### train.py
Basic feature extraction using max, mean, and standard deviation.

### train2.py
Advanced feature extraction with peak counting:
- Counts threshold crossings to detect blink patterns
- Configurable `PEAK_THRESHOLD` parameter
- Higher accuracy for multi-blink detection

## Telegram Setup

1. Create a bot via [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
4. Update credentials in firmware

## Troubleshooting

**WiFi Connection Issues**
- Verify SSID and password
- Check 2.4GHz network availability
- Monitor Serial output (115200 baud)

**Low Accuracy**
- Collect more training data
- Adjust `PEAK_THRESHOLD` in train2.py
- Ensure consistent blink patterns during data collection

**Telegram Alerts Not Sending**
- Verify bot token and chat ID
- Check internet connectivity
- Review Serial monitor for HTTP error codes

## Performance

- **Sampling Rate**: 100 Hz
- **Prediction Latency**: ~1 second
- **Model Size**: <2KB (embedded in firmware)
- **Accuracy**: Varies by training data quality (typically >85%)

## License

Open source - feel free to modify and adapt for your needs.

## Future Enhancements

- Support for more blink patterns
- Adaptive threshold calibration
- Battery-powered operation
- Mobile app integration
- Multi-user support
