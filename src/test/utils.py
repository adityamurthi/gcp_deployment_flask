import torch
import torch.nn as nn

from src.test.model import LSTMRegression

# Get the device name string.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(path_to_data: str) -> torch.Tensor:

    """
    Loads the data from a particular path

    :param path_to_data: str -> Full filename path of data.
    :return:
    """
    # Get a single slice of data
    data = torch.load(path_to_data)[0][0]
    # Cast data as  3D tensor because the model was trained on batches.
    return data.view(-1, data.shape[0], data.shape[1])


def load_model(path_to_model: str):
    """
    Loads the model from local disk. This can be set to load when the app runs from a GCP bucket.
    :param path_to_model: str -> Full file apth to model
    :return:
    """

    # Load the state_dict (Load the model trained on a GPU onto a CPU)
    saved_state_dict = torch.load(path_to_model, map_location=device)
    print(saved_state_dict[1])
    num_inputs = saved_state_dict[1]['n_inputs']
    num_hidden = saved_state_dict[1]['num_hidden']
    num_layers = saved_state_dict[1]['num_layers']

    model_lstm = LSTMRegression(input_size=num_inputs, n_hidden=num_hidden, n_layers=num_layers)
    model_lstm.load_state_dict(saved_state_dict[0])

    return model_lstm


def get_predictions(model: nn.Module, data: torch.Tensor) -> float:
    """The main method used to get the predictions from input data"""
    with torch.no_grad():
        model.eval()
        prediction = model(data)
    return prediction.item()


if __name__ == "__main__":
    path_to_data = '/Users/Adi/ML/gcp_deployment_flask/data/bike_rides_test_sample.pt'
    # Load the data
    data = load_data(path_to_data)
    print(data)
    # Load the model
    path_to_model = '/Users/Adi/ML/gcp_deployment_flask/model/lstm_checkpoint_nn_2_sl_24_epoch_41.pt'
    model = load_model(path_to_model)
    print(model)
    print(get_predictions(model, data))






