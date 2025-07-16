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

@app.route('/predict_class', methods=['POST'])
def predict_class():
    data = request.get_json()
    X = np.array(data["X"])  # Shape: [[f1, f2, ...]]
    preds = manager.predict(X)  # NumPy array like [0]
    return jsonify({"prediction": int(preds[0])})


@app.route('/set_model', methods=['POST'])
def set_model():
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
