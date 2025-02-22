"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
Author: Patrick Emamim, Chi Zhang, Modified by Jet Chiang
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple

from .eiie import CNNPredictor, LSTMPredictor

def create_target_network(model: nn.Module):
    """Create a target network from the model"""
    target = copy.deepcopy(model)
    for param in target.parameters():
        param.requires_grad = False
    return target

class StockActor(nn.Module):
    """Actor network for DDPG, responsible for action selection"""
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        super(StockActor, self).__init__()
        self.s_dim = state_dim      # e.g., [nb_classes, window_length, 4] -> preprocess to only one feature
        self.a_dim = action_dim      # e.g., [nb_actions]
        self.action_bound = action_bound
        self.tau = tau
        self.batch_size = batch_size
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm

        if predictor_type == 'cnn':
            self.predictor = CNNPredictor(input_dim=state_dim, output_dim=(1, 1), use_batch_norm=use_batch_norm)
        elif predictor_type == 'lstm':
            self.predictor = LSTMPredictor(input_dim=state_dim, output_dim=(1, 1), hidden_dim=64, use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Predictor type not recognized')
        
        layers = []
        layers.append(nn.Linear(self.s_dim[0]*32, 64))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 64))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, action_dim[0]))
        torch.nn.init.uniform_(layers[-1].weight, a=-0.003, b=0.003)

        self.fc_layers = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.target_network = create_target_network(self)

    def forward(self, state):
        """
        Forward pass with gradient tracking
        Scales the output to the action bound
        using a softmax activation function
        """
        x = self.predictor(state)
        x = self.fc_layers(x)
        # softmax ensure the output is between 0 and 1
        x = torch.softmax(x, dim=-1)
        # scale the output to the action bound (portfolio weight)
        scaled_out = x * self.action_bound
        return scaled_out

    def train_step(self, inputs, critic) -> Tuple[torch.Tensor, float]:
        """Train the actor network by maximizing the Q value
        Args:
            inputs (torch.Tensor): input tensor
            critic (StockCritic): critic network
        """
        self.optimizer.zero_grad()
        actions = self.forward(inputs)
        # predict the Q value using the critic network
        q_values = critic.predict(inputs, actions)
        loss = -torch.mean(q_values)
        loss.backward()
        self.optimizer.step()
        return q_values, loss.item()

    def predict(self, inputs):
        """Predict the action given the input"""
        self.eval()
        with torch.no_grad():
            actions = self.forward(inputs)
        self.train()
        return actions

    def predict_target(self, inputs):
        """
        Predict the action given the input using the target network
        This differs from the predict method in that it uses the target network
        """
        self.eval()
        with torch.no_grad():
            actions = self.target_network(inputs)
            actions = torch.tanh(actions) * self.action_bound
        self.train()
        return actions

    def update_target_network(self):
        """Update the target network using the current network"""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)
