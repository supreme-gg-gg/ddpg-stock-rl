"""
Contain an abstract base model that all the subclass need to follow the API
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAgent(ABC, object):
    """Abstract base class for different agents (DDPG, PPO, etc.)"""
    @abstractmethod
    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length, num_features).
            Feature contains (open, high, low, close)

        Returns: action to take at next timestamp. A numpy array shape (num_stocks + 1,)

        """
        raise NotImplementedError('This method must be implemented by subclass')

class PredictorBase(nn.Module, ABC):
    """Abstract base class for different predictors (CNN, LSTM, etc.)"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass")
    
class FullyConnectedLayers(nn.Module):
    """Final fully connected layers for the actor and critic"""
    def __init__(self, input_dim, output_dim, use_batch_norm):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        self.fc3 = nn.Linear(64, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x

class BaseNetwork(nn.Module, ABC):
    """Abstract base class for Actor and Critic"""
    def __init__(self, predictor: PredictorBase, fc_layers: FullyConnectedLayers):
        super().__init__()
        self.predictor = predictor
        self.fc_layers = fc_layers

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass")
