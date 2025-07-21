# ml_backend/api.py
from flask import Flask, request, jsonify
from model_manager import ModelManager
import numpy as np
import json
import uuid
import os

from db import init_db, Session, APIKey
init_db()
app = Flask(__name__)

# -------------------------------
# Persistent API key system
# -------------------------------
from db import Session, APIKey

def check_api_key():
    key = request.headers.get("X-API-Key")
    session = Session()
    valid = session.query(APIKey).filter_by(api_key=key).first() is not None
    session.close()
    return valid


@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.get_json()
    username = data.get("username")
    if not username:
        return jsonify({"error": "Username is required"}), 400

    session = Session()
    existing = session.query(APIKey).filter_by(username=username).first()
    if existing:
        session.close()
        return jsonify({"message": "User already exists", "api_key": existing.api_key})

    api_key = str(uuid.uuid4())[:12]
    new_user = APIKey(username=username, api_key=api_key)
    session.add(new_user)
    session.commit()
    session.close()

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
