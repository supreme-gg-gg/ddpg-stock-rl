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

    The input shape will be [batch_size * num_stock, window_length, num_features]
    The output shape will be [batch_size, num_stock * hidden_dim]
    """
    def __init__(self, input_dim: tuple, output_dim: tuple, hidden_dim: int, use_batch_norm: bool):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.window_length = input_dim[1]
        self.num_stocks = input_dim[0]
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        x = x.view(-1, self.window_length, 1) # reshape to [batch_size * num_stock, window_length, 1]
        x, _ = self.lstm(x)
        x = x.view(-1, self.hidden_dim * self.num_stocks) # reshape to [batch_size, num_stock * hidden_dim]
        return x