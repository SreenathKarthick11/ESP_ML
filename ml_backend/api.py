# ml_backend/api.py
from flask import Flask, request, jsonify
from model_manager import ModelManager
import numpy as np
import json
import uuid
import os

app = Flask(__name__)

# -------------------------------
# Persistent API key system
# -------------------------------
KEY_FILE = "api_keys.json"

def load_keys():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_keys(keys):
    with open(KEY_FILE, "w") as f:
        json.dump(keys, f)

API_KEYS = load_keys()

def check_api_key():
    key = request.headers.get("X-API-Key")
    return key in API_KEYS.values()

@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get("username")
    if not username:
        return jsonify({"error": "Username is required"}), 400
    if username in API_KEYS:
        return jsonify({"message": "User already exists", "api_key": API_KEYS[username]})
    
    api_key = str(uuid.uuid4())[:12]
    API_KEYS[username] = api_key
    save_keys(API_KEYS)
    return jsonify({"username": username, "api_key": api_key})

# Initialize with logistic regression by default
manager = ModelManager(model_type='logistic')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    X = np.array(data['X'])  # Expecting [[f1, f2, ...], ...]
    y = np.array(data['y'])  # Expecting [label1, label2, ...]

    manager.train(X, y)
    return jsonify({"message": "Training complete!"})

@app.route('/get_weights', methods=['GET'])
def get_weights():
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    weights, bias = manager.get_weights()
    if weights is not None:
        return jsonify({"weights": weights, "bias": bias})
    else:
        return jsonify({"message": "No weights available for this model."})

@app.route('/ping', methods=['GET'])
def ping():
    return "Server alive!"

@app.route('/predict_class', methods=['POST'])
def predict_class():
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    X = np.array(data["X"])  # Shape: [[f1, f2, ...]]
    preds = manager.predict(X)  # NumPy array like [0]
    return jsonify({"prediction": int(preds[0])})


@app.route('/set_model', methods=['POST'])
def set_model():
    if not check_api_key():
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    model_type = data.get("model_type", "logistic")
    max_depth = data.get("max_depth", 3)
    try:
        manager.set_model(model_type, max_depth)
        return jsonify({"message": f"Model set to {model_type} with depth={max_depth}"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
