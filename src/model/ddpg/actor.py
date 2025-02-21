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

from base import BaseNetwork, FullyConnectedLayers
from eiie import CNNPredictor, LSTMPredictor

def create_target_network(model: nn.Module):
    """Create a target network from the model"""
    target = copy.deepcopy(model)
    for param in target.parameters():
        param.requires_grad = False
    return target

class StockActor(BaseNetwork):
    """Actor network for DDPG, responsible for action selection"""
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        super(StockActor, self).__init__()
        self.s_dim = state_dim      # e.g., [batch_size, nb_classes, window_length, 4]
        self.a_dim = action_dim      # e.g., [batch_size, nb_actions]
        self.action_bound = action_bound
        self.tau = tau
        self.batch_size = batch_size
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm

        if predictor_type == 'cnn':
            predictor = CNNPredictor(input_dim=(1, state_dim[1], state_dim[3]), output_dim=(1, 1), use_batch_norm=use_batch_norm)
        elif predictor_type == 'lstm':
            predictor = LSTMPredictor(input_dim=(state_dim[1], state_dim[3]), output_dim=(1, 1), hidden_dim=64, use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Predictor type not recognized')
        
        fc_layers = FullyConnectedLayers(input_dim=64, output_dim=action_dim[1], use_batch_norm=use_batch_norm)
        super().__init__(predictor, fc_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.target_network = create_target_network(self)

    def forward(self, state):
        x = self.predictor(state)
        x = self.fc_layers(x)
        return torch.tanh(x) * self.action_bound

    def train_step(self, inputs, a_gradient):
        self.optimizer.zero_grad()
        actions = self.forward(inputs)
        loss = -torch.mean(actions * a_gradient)
        loss.backward()
        self.optimizer.step()

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            actions = self.forward(inputs)
        self.train()
        return actions

    def predict_target(self, inputs):
        self.eval()
        with torch.no_grad():
            actions = self.target_network(inputs)
            actions = torch.tanh(actions) * self.action_bound
        self.train()
        return actions

    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)