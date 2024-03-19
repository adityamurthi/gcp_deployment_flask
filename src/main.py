from flask import Flask, request, jsonify

import torch.nn as nn
import torch

import pickle
import json

# The model is stored in the source directory
MODEL_NAME = 'lstm_checkpoint_nn_2_sl_24_epoch_41.pt'

# Get the device name string.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constructor to use "main" as the app.
app = Flask(__name__)


class LSTMRegression(nn.Module):
  def __init__(self, input_size: int, n_hidden: int, n_layers: int = 1, dropout_p: float = 0.2):
    super(LSTMRegression, self).__init__()
    self.num_layers = n_layers
    self.hidden_size = n_hidden

    if self.num_layers > 1:
      self.lstm = nn.LSTM(
          input_size=input_size,
          hidden_size=self.hidden_size,
          num_layers=self.num_layers,
          batch_first=True,
          dropout=dropout_p
          )
    else:
      self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        batch_first=True
        )

    # Fully connected linear layer that is set to output a sigle value
    self.regressor = nn.Linear(self.hidden_size, 1)

  def forward(self, x):
    # Improves memory allocation during distributed training
    self.lstm.flatten_parameters()
    batch_size = x.shape[0]
    # Initialize the hidden and cell state tensors to zero (although the nn.LSTM takes care of this!)
    h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(self.num_layers, batch_size,  self.hidden_size).to(device)

    # propagate the input through the LSTM: You will get output, (hn, cn)
    out, _ = self.lstm(x, (h0, c0))

    # Feed the hidden layer to the linear layer to get the prediction.
    output = self.regressor(out[:, -1, :])
    return output


def load_model(model_name: str = MODEL_NAME):
    """
    Loads the model from local disk. This can be set to load when the app runs from a GCP bucket.
    :param model_name: str -> Full filename of saved model
    :return:
    """

    # Load the state_dict (Load the model trained on a GPU onto a CPU)
    saved_state_dict = torch.load(model_name, map_location=device)
    num_inputs = saved_state_dict[1]['n_inputs']
    num_hidden = saved_state_dict[1]['num_hidden']
    num_layers = saved_state_dict[1]['num_layers']

    model_lstm = LSTMRegression(input_size=num_inputs, n_hidden=num_hidden, n_layers=num_layers)
    model_lstm.load_state_dict(saved_state_dict[0])

    return model_lstm


# Load model from saved model file.
model = load_model()


def get_predictions(model: nn.Module, data: torch.Tensor) -> float:
    """The main method used to get the predictions from input data"""
    with torch.no_grad():
        model.eval()
        prediction = model(data)
    return prediction.item()


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
