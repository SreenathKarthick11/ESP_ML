// esp32_ml_client.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_ml_client.h"
#include "cJSON.h"
#include "esp_http_client.h"

#define MAX_SAMPLES 100
#define MAX_FEATURES 10

static float X_buffer[MAX_SAMPLES][MAX_FEATURES];
static float y_buffer[MAX_SAMPLES];
static int current_index = 0;

static float weights[MAX_FEATURES];
static float bias = 0.0f;
static int max_features = MAX_FEATURES;

void esp_ml_init(int num_features) {
    if (num_features > MAX_FEATURES) {
        num_features = MAX_FEATURES;
    }
    for (int i = 0; i < num_features; i++) {
        weights[i] = 0.0f;
    }
    bias = 0.0f;
    current_index = 0;
}

void append_sample(float *features, float label, int n_features) {
    if (current_index >= MAX_SAMPLES) return;
    for (int i = 0; i < n_features; i++) {
        X_buffer[current_index][i] = features[i];
    }
    y_buffer[current_index] = label;
    current_index++;
}

char *create_json_payload(int n_features, int n_samples) {
    cJSON *root = cJSON_CreateObject();
    cJSON *X_array = cJSON_CreateArray();
    cJSON *y_array = cJSON_CreateArray();

    for (int i = 0; i < n_samples; i++) {
        cJSON *sample = cJSON_CreateArray();
        for (int j = 0; j < n_features; j++) {
            cJSON_AddItemToArray(sample, cJSON_CreateNumber(X_buffer[i][j]));
        }
        cJSON_AddItemToArray(X_array, sample);
        cJSON_AddItemToArray(y_array, cJSON_CreateNumber(y_buffer[i]));
    }

    cJSON_AddItemToObject(root, "X", X_array);
    cJSON_AddItemToObject(root, "y", y_array);

    char *json_str = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return json_str;
}

esp_err_t send_data_to_server(const char *json_payload, const char *server_url) {
    esp_http_client_config_t config = {
        .url = server_url,
        .method = HTTP_METHOD_POST,
    };
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_http_client_set_header(client, "Content-Type", "application/json");
    esp_http_client_set_post_field(client, json_payload, strlen(json_payload));

    esp_err_t err = esp_http_client_perform(client);
    esp_http_client_cleanup(client);
    return err;
}

float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

float predict(float *x, float *w, float bias, int n_features) {
    float z = 0.0f;
    for (int i = 0; i < n_features; i++) {
        z += x[i] * w[i];
    }
    z += bias;
    return sigmoid(z);
}

void update_weights(float *new_weights, float new_bias, int n_features) {
    for (int i = 0; i < n_features; i++) {
        weights[i] = new_weights[i];
    }
    bias = new_bias;
}
