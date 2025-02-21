"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
Author: Patrick Emami, Modified by Chi Zhang
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# Assume stock_predictor is already implemented and imported
from util import create_predictor_model

# --- Old TensorFlow implementation (preserved) ---
"""
"""
# """
# Actor Network definition, The CNN architecture follows the one in this paper
# https://arxiv.org/abs/1706.10059
# Author: Patrick Emami, Modified by Chi Zhang
#
# import tensorflow as tf
# import tflearn
#
# class ActorNetwork(object):
#     def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
#         # ...existing code...
#         self.inputs, self.out, self.scaled_out = self.create_actor_network()
#         self.network_params = tf.trainable_variables()
#         # ...existing code...
#         self.action_gradient = tf.placeholder(tf.float32, [None] + self.a_dim)
#         # ...existing code...
#         self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))
#         # ...existing code...
#
#     def create_actor_network(self):
#         raise NotImplementedError('Create actor should return (inputs, out, scaled_out)')
#
#     def train(self, inputs, a_gradient):
#         # ...existing code...
#         self.sess.run(self.optimize, feed_dict={ self.inputs: inputs, self.action_gradient: a_gradient })
#
#     def predict(self, inputs):
#         return self.sess.run(self.scaled_out, feed_dict={ self.inputs: inputs })
#
#     def update_target_network(self):
#         # ...existing code...
#         for target_param, param in zip(self.target_network_params, self.network_params):
#             target_param.data.copy_(target_param.data * self.tau + param.data * (1. - self.tau))
# 
# class StockActor(ActorNetwork):
#     def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size,
#                  predictor_type, use_batch_norm):
#         # ...existing code...
#
#     def create_actor_network(self):
#         # ...existing code...
#         inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')
#         net = stock_predictor(inputs, self.predictor_type, self.use_batch_norm)
#         # ...existing code...
#         return inputs, out, scaled_out
#
#     def train(self, inputs, a_gradient):
#         # ...existing code...
#         self.sess.run(self.optimize, feed_dict={ self.inputs: inputs, self.action_gradient: a_gradient })
#
#     def predict(self, inputs):
#         # ...existing code...
#         return self.sess.run(self.scaled_out, feed_dict={ self.inputs: inputs })
# """
# --- End of old implementation ---


# New coherent PyTorch implementation combined into StockActor
class StockActor(nn.Module):
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

        # Build network using stock_predictor and additional FC layers
        predictor = create_predictor_model(self.s_dim, self.predictor_type, self.use_batch_norm)
        
        # Calculate predictor output dimension
        if predictor_type == 'cnn':
            # For CNN: num_filters * num_stocks
            self.predictor_out_dim = 32 * state_dim[0]  # 32 is the number of filters
        else:
            # For LSTM: hidden_dim * num_stocks
            self.predictor_out_dim = 32 * state_dim[0]  # 32 is the hidden_dim
        
        fc_layers = nn.Sequential(
            nn.Linear(self.predictor_out_dim, 64),
            nn.BatchNorm1d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64) if self.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, self.a_dim[0]),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.target_network = copy.deepcopy(self)

    def forward(self, x):
        # Process input as in original (e.g., flatten if needed)
        # x = x[:, :, -self.s_dim[1]:, :]  # uncomment if slicing is required
        x = x.view(x.size(0), -1)
        out = self.network(x)
        return out * self.action_bound

    def train_step(self, inputs, a_gradient):
        self.optimizer.zero_grad()
        inputs = inputs.view(inputs.size(0), -1)
        actions = self.forward(inputs)
        # Loss is defined to maximize the expected value (i.e. minimize negative)
        loss = -torch.mean(actions * a_gradient)
        loss.backward()
        self.optimizer.step()

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            inputs = inputs.view(inputs.size(0), -1)
            actions = self.forward(inputs)
        self.train()
        return actions

    def predict_target(self, inputs):
        self.eval()
        with torch.no_grad():
            inputs = inputs.view(inputs.size(0), -1)
            out = self.target_network(inputs)
            actions = out * self.action_bound
        self.train()
        return actions

    def update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1. - self.tau) * target_param.data)