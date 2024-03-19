# Using PyTorch
import torch
import torch.nn as nn

# Get the device name string.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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