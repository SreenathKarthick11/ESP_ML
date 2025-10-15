# ESP_ML

**ESP_ML** is a lightweight **client–server framework** designed to enable **real-time sensor data collection** from ESP32 microcontrollers and **seamless communication with cloud-hosted machine learning models** via REST APIs.

It bridges the gap between **embedded IoT systems** and **cloud-based ML inference**, providing a scalable foundation for intelligent IoT applications.

**Project Website:** [https://esp-ml-backend.onrender.com/](Live)

---

## Overview

ESP_ML consists of two core components:

1. **ESP32 Client Library (C/C++)**

   * Collects real-time sensor data.
   * Communicates with the backend ML inference API using HTTP/REST.

2. **ML Backend Server (Python + Flask)**

   * Hosts trained ML models.
   * Manages incoming sensor data and performs inference.
   * Sends results back to connected ESP32 clients.

---

## Project Structure

```
ESP_ML/
├── esp_ml_client/           # ESP32 client source code (C/C++)
│   ├── esp_ml_client.cpp
│   ├── esp_ml_client.h
│   └── library.properties
│
├── ml_backend/              # Python backend service
│   ├── api.py               # Flask API endpoints
│   ├── db.py                # Local/SQLite database handler
│   ├── model_manager.py     # Model loading & inference logic
│   ├── utils.py             # Helper utilities
│   ├── requirements.txt     # Python dependencies
│   ├── templates/           # Web UI templates
│   ├── static/              # CSS, images, and client zip
│   └── keys.db              # Local key store (optional)
│
└── README.md
```

---

## Installation & Setup

### Backend (Flask ML Server)

#### **Prerequisites**

* Python 3.10+
* `pip` package manager

#### **Setup**

```bash
cd ESP_ML/ml_backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### **Run the Server**

```bash
python api.py
```

> Default address: [http://127.0.0.1:5000](http://127.0.0.1:5000)
> You can host the backend on a **cloud VM**, **Raspberry Pi**, or any REST-accessible environment.

---

### ESP32 Client

#### **Prerequisites**

* Arduino IDE or PlatformIO
* ESP32 board package installed
* Wi-Fi network credentials
* URL of the ML backend server

#### **Integration Steps**

1. Include the ESP_ML client library in your ESP32 project.
2. Initialize the client with your Wi-Fi credentials and backend URL.
3. Send sensor data and receive inference results using the provided API.

#### **Example Sketch**

```cpp
#include "esp_ml_client.h"

ESPMLClient client("your_ssid", "your_password", "http://<server_ip>:5000");

void setup() {
  Serial.begin(115200);
  client.connectWiFi();
}

void loop() {
  float sensorValue = analogRead(34);
  String result = client.sendData(sensorValue);
  Serial.println("Prediction: " + result);
  delay(2000);
}
```

---

## Features

✅ Real-time sensor data acquisition \
✅ Lightweight REST communication for embedded systems \
✅ Cloud-hosted ML inference via Flask \
✅ Modular architecture — replaceable backend or model \
✅ Easy ESP32 integration \
✅ Optional visualization dashboard (HTML templates) \

---

## Backend API Endpoints

| Endpoint        | Method | Description                                        |
| --------------- | ------ | -------------------------------------------------- |
| `/predict`      | POST   | Receives JSON sensor data and returns ML inference |
| `/register`     | POST   | Registers a new device/client                      |
| `/status`       | GET    | Health check endpoint                              |
| `/upload_model` | POST   | Uploads or replaces an ML model (admin use)        |

#### **Example Request**

```json
POST /predict
{
  "device_id": "esp32_01",
  "data": [0.21, 0.34, 0.56, 0.12]
}
```

#### **Example Response**

```json
{
  "prediction": "normal",
  "confidence": 0.92
}
```

---

## Dependencies

### **Backend (Python)**

Listed in `ml_backend/requirements.txt`:

```
Flask
numpy
pandas
scikit-learn
requests
```

*(Dependencies may vary depending on your model setup.)*

### **ESP32 Client (C/C++)**

* Arduino Core for ESP32
* WiFiClient / HTTPClient (built-in)
* ArduinoJson (if used)

---

## Configuration

| Component | Parameter          | Description              |
| --------- | ------------------ | ------------------------ |
| Client    | `SSID`, `PASSWORD` | Wi-Fi credentials        |
| Client    | `API_URL`          | URL of backend ML API    |
| Backend   | `MODEL_PATH`       | Path to trained ML model |
| Backend   | `PORT`             | Flask server port        |

---

## Example Workflow

1. Deploy Flask backend (local/cloud).
2. Connect ESP32 to Wi-Fi and set backend URL.
3. ESP32 sends real-time sensor readings to the API.
4. Backend performs ML inference (e.g., anomaly detection).
5. ESP32 receives and displays prediction results.

---

## Troubleshooting

| Issue                         | Cause                          | Solution                                |
| ----------------------------- | ------------------------------ | --------------------------------------- |
| ESP32 not connecting to Wi-Fi | Incorrect SSID/PASSWORD        | Verify credentials                      |
| No response from server       | Wrong API URL or port          | Check Flask server IP and port          |
| Connection refused            | Server inactive/firewall issue | Ensure Flask is running, port 5000 open |
| Unexpected inference output   | Model or data mismatch         | Verify JSON format and input schema     |

---

## Contributor

**M Sreenath Karthick** — Developer

Contributions are welcome!
Feel free to open issues or submit pull requests to improve the project.


