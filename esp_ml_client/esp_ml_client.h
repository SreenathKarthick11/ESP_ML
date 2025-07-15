#ifndef ESP_ML_Client_h
#define ESP_ML_Client_h

#include <Arduino.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>

class ESP_ML_Client {
  public:
    ESP_ML_Client(int n_features, int max_samples = 100);
    ~ESP_ML_Client();
    void appendSample(float *features, float label);
    String createJsonPayload();
    bool sendDataToServer(const char *server_url, const String &json_payload);
    bool getWeights(const char *server_url, float *weights, float &bias);
    float predict(float *features, float *weights, float bias);

  private:
    int n_features;
    int max_samples;
    int current_index;
    float **X_buffer;
    float *y_buffer;
};

#endif
