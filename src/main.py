from flask import Flask, request, jsonify

import torch.nn as nn
import torch

import pickle
import json

from src.test.utils import load_model, get_predictions

# Constructor to use "main" as the app.
app = Flask(__name__)

DATA_PATH = '/Users/Adi/ML/gcp_deployment_flask/data/bike_rides_test_sample.pt'
MODEL_PATH = '/Users/Adi/ML/gcp_deployment_flask/model/lstm_checkpoint_nn_2_sl_24_epoch_41.pt'

# Get the device name string.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model from saved model file.
model = load_model(MODEL_PATH)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Main method on the server side that is called
    to get predictions
    """
    # This requests the client to provide the data as a serialized JSON object
    serialized_data = request.get_json()
    input_tensor = pickle.loads(json.loads(serialized_data).encode('latin-1'))

    try:
        isinstance(input_tensor, torch.Tensor)
    except Exception as e:
        return jsonify({"error": e})

    return jsonify({'prediction': get_predictions(model, input_tensor)})


if __name__ == "__main__":
    app.run(debug=True)
