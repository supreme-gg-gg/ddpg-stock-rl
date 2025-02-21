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
