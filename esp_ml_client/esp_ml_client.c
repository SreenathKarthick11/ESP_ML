// esp32_ml_client/esp_ml_client.c

#include <math.h>
#include "esp_ml_client.h"

static float weights[10]; // Example: max 10 features
static float bias = 0.0f;

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
