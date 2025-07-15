// esp32_ml_client/esp_ml_client.h

#ifndef ESP_ML_CLIENT_H
#define ESP_ML_CLIENT_H

float sigmoid(float z);
float predict(float *x, float *w, float bias, int n_features);
void update_weights(float *new_weights, float new_bias, int n_features);

#endif
