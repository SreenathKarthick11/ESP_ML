# ml_backend/api.py
from flask import Flask, request, jsonify
from model_manager import ModelManager
import numpy as np

app = Flask(__name__)

# Initialize with logistic regression by default
manager = ModelManager(model_type='logistic')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    data = request.get_json()
    X = np.array(data['X'])  # Expecting [[f1, f2, ...], ...]
    y = np.array(data['y'])  # Expecting [label1, label2, ...]

    manager.train(X, y)
    return jsonify({"message": "Training complete!"})

@app.route('/get_weights', methods=['GET'])
def get_weights():
    weights, bias = manager.get_weights()
    if weights is not None:
        return jsonify({"weights": weights, "bias": bias})
    else:
        return jsonify({"message": "No weights available for this model."})

@app.route('/ping', methods=['GET'])
def ping():
    return "Server alive!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
