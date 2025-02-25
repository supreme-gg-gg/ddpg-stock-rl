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
        self.s_dim = state_dim      # e.g., [nb_classes, window_length, 4] -> preprocess to only one feature hence [nb_classes, window_length]
        self.a_dim = action_dim      # e.g., [nb_actions]
        self.action_bound = action_bound
        self.tau = tau
        self.batch_size = batch_size
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm

        self.lstm_hidden_dim = 64

        if predictor_type == 'cnn':
            self.predictor = CNNPredictor(input_dim=state_dim, output_dim=(1, 1), use_batch_norm=use_batch_norm)
        elif predictor_type == 'lstm':
            self.predictor = LSTMPredictor(input_dim=state_dim, output_dim=(1, 1), hidden_dim=self.lstm_hidden_dim, use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Predictor type not recognized')
        
        layers = []
        layers.append(nn.Linear(self.s_dim[0] * self.lstm_hidden_dim, 64))
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

        for param in critic.parameters():
            param.requires_grad = False

        q_values = critic.forward(inputs, actions)
        loss = -torch.mean(q_values)
        loss.backward()
        self.optimizer.step()

        for param in critic.parameters():
            param.requires_grad = True

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
        self.target_network.eval()
        with torch.no_grad():
            actions = self.target_network(inputs)
            # actions = torch.tanh(actions) * self.action_bound
        self.target_network.train()
        return actions

    def update_target_network(self):
        """Update the target network using the current network"""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)


class StockActorPVM(StockActor):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm):
        super(StockActorPVM, self).__init__(state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
                 predictor_type, use_batch_norm)
        
        # concatenate the weights
        self.lstm_hidden_dim = 16
        self.predictor = LSTMPredictor(input_dim=state_dim, output_dim=(1, 1), hidden_dim=self.lstm_hidden_dim, use_batch_norm=use_batch_norm)
        # self.fc_layers[0] = nn.Linear(self.s_dim[0] * (self.lstm_hidden_dim + 1), 64)
        # self.target_network.fc_layers[0] = self.fc_layers[0]

        self.conv1d = nn.Conv1d(in_channels=self.lstm_hidden_dim + 1, out_channels=32, kernel_size=1)
        self.fc1 = nn.Linear(32 * self.s_dim[0], 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, self.s_dim[0])
        # self.target_network.conv1d = self.conv1d
        # self.target_network.fc1 = self.fc1
        # self.target_network.bn1 = self.bn1
        # self.target_network.fc2 = self.fc2
        self.target_network = create_target_network(self)


    # def forward(self, state, weights):
    #     x = self.predictor(state)
    #     x = x.view(-1, self.s_dim[0] * self.lstm_hidden_dim)
    #     x = torch.cat((x, weights), dim=1)
    #     x = self.fc_layers(x)
    #     # softmax ensure the output is between 0 and 1
    #     x = torch.softmax(x, dim=-1)
    #     # scale the output to the action bound (portfolio weight)
    #     scaled_out = x * self.action_bound
    #     return scaled_out
    
    def forward(self, state, weights):
        batch_size = state.size(0)
        lstm_out = self.predictor(state)
        lstm_out = lstm_out.view(-1, self.s_dim[0], self.lstm_hidden_dim)

        prev_w = weights.unsqueeze(-1)
        x = torch.cat([lstm_out, prev_w], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = F.relu(x)
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)  # shape: [batch_size, num_stocks]
        x = torch.softmax(x, dim=-1)
        
        scaled_out = x * self.action_bound
        return scaled_out
    
    def predict(self, state, weights):
        """Predict the action given the input"""
        self.eval()
        with torch.no_grad():
            actions = self.forward(state, weights)
        self.train()
        return actions
    
    def predict_target(self, state, weights):
        """
        Predict the action given the input using the target network
        This differs from the predict method in that it uses the target network
        """
        self.target_network.eval()
        with torch.no_grad():
            actions = self.target_network(state, weights)
            # actions = torch.tanh(actions) * self.action_bound
        self.target_network.train()
        return actions
    
    def train_step(self, state, weights, critic) -> Tuple[torch.Tensor, float]:
        """Train the actor network by maximizing the Q value
        Args:
            inputs (torch.Tensor): input tensor
            critic (StockCritic): critic network
        """
        self.optimizer.zero_grad()
        actions = self.forward(state, weights)
        # predict the Q value using the critic network

        for param in critic.parameters():
            param.requires_grad = False

        q_values = critic.forward(state, weights, actions)
        loss = -torch.mean(q_values)
        loss.backward()
        self.optimizer.step()

        for param in critic.parameters():
            param.requires_grad = True

        return q_values, loss.item()