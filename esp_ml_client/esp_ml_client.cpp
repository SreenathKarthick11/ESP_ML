#include "ESP_ML_Client.h"

ESP_ML_Client::ESP_ML_Client(int n_features, int max_samples) {
  this->n_features = n_features;
  this->max_samples = max_samples;
  this->current_index = 0;

  X_buffer = new float*[max_samples];
  for (int i = 0; i < max_samples; i++) {
    X_buffer[i] = new float[n_features];
  }
  y_buffer = new float[max_samples];
}

ESP_ML_Client::~ESP_ML_Client() {
  for (int i = 0; i < max_samples; i++) {
    delete[] X_buffer[i];
  }
  delete[] X_buffer;
  delete[] y_buffer;
}

void ESP_ML_Client::appendSample(float *features, float label) {
  if (current_index >= max_samples) return;
  for (int i = 0; i < n_features; i++) {
    X_buffer[current_index][i] = features[i];
  }
  y_buffer[current_index] = label;
  current_index++;
}

String ESP_ML_Client::createJsonPayload() {
  StaticJsonDocument<8192> doc;
  JsonArray X = doc.createNestedArray("X");
  JsonArray y = doc.createNestedArray("y");

  for (int i = 0; i < current_index; i++) {
    JsonArray sample = X.createNestedArray();
    for (int j = 0; j < n_features; j++) {
      sample.add(X_buffer[i][j]);
    }
    y.add(y_buffer[i]);
  }

  String output;
  serializeJson(doc, output);
  return output;
}

bool ESP_ML_Client::sendDataToServer(const char *server_url, const String &json_payload) {
  HTTPClient http;
  http.begin(server_url);
  http.addHeader("Content-Type", "application/json");
  int httpResponseCode = http.POST(json_payload);

  http.end();
  return httpResponseCode > 0;
}

bool ESP_ML_Client::getWeights(const char *server_url, float *weights, float &bias) {
  HTTPClient http;
  http.begin(server_url);
  int httpResponseCode = http.GET();

  if (httpResponseCode > 0) {
    String response = http.getString();
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, response);

    if (!error) {
      JsonArray w_array = doc["weights"].as<JsonArray>();
      for (int i = 0; i < n_features; i++) {
        weights[i] = w_array[i];
      }
      bias = doc["bias"];
      http.end();
      return true;
    }
  }
  http.end();
  return false;
}

float ESP_ML_Client::predict(float *features, float *weights, float bias) {
  float z = 0.0f;
  for (int i = 0; i < n_features; i++) {
    z += features[i] * weights[i];
  }
  z += bias;
  return 1.0 / (1.0 + exp(-z));
}
