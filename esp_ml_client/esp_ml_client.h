// esp32_ml_client.h
#ifndef ESP_ML_CLIENT_H
#define ESP_ML_CLIENT_H

#include "esp_err.h"

void esp_ml_init(int num_features);
void append_sample(float *features, float label, int n_features);
char *create_json_payload(int n_features, int n_samples);
esp_err_t send_data_to_server(const char *json_payload, const char *server_url);

float sigmoid(float z);
float predict(float *x, float *w, float bias, int n_features);
void update_weights(float *new_weights, float new_bias, int n_features);

#endif