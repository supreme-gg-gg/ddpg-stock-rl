import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import PredictorBase
    
class CNNPredictor(PredictorBase):
    """
    CNN predictor model

    The input shape will be [batch_size, num_stock, window_length, num_features]
    The output shape will be [batch_size, num_stock * num_filters]
    """
    def __init__(self, input_dim: tuple, output_dim: tuple, use_batch_norm: bool):
        super().__init__(input_dim, output_dim)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 3))

        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(32)
        else:
            self.bn1 = lambda x: x

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, input_dim[2] - 2))

        if use_batch_norm:
            self.bn2 = nn.BatchNorm2d(32)
        else:
            self.bn2 = lambda x: x

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.flatten(x)
        return x
    
class LSTMPredictor(PredictorBase):
    """
    LSTM predictor model

    Expected input shape: [batch_size, num_stocks, window_length, num_features]
    Output shape: [batch_size, num_stocks * hidden_dim]
    """
    def __init__(self, input_dim: tuple, output_dim: tuple, hidden_dim: int, use_batch_norm: bool):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        # input_dim should be (num_stocks, window_length, num_features)
        self.num_stocks = input_dim[0]
        self.window_length = input_dim[1]
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        # x is expected to have shape: [batch_size, num_stocks, window_length, 1]
        batch_size = x.shape[0]
        num_stocks = x.shape[1] # should equal self.num_stocks
        window_length = x.shape[2]
        
        # Reshape to combine batch and stocks:
        # New shape: [batch_size * num_stocks, window_length, 1]
        x = x.view(batch_size * num_stocks, window_length, 1)
        
        # Pass through LSTM. PyTorch LSTM returns the full sequence; we take the last time step.
        x, _ = self.lstm(x)  # shape: [batch_size * num_stocks, window_length, hidden_dim]
        x = x[:, -1, :]      # shape: [batch_size * num_stocks, hidden_dim]
        # Flatten the last two dimensions to obtain [batch_size, num_stocks * hidden_dim]
        return x