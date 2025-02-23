from .eiie import CNNPredictor, LSTMPredictor

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# It is assumed that these predictors are defined elsewhere:
# from eiie import CNNPredictor, LSTMPredictor

def create_target_network(model: nn.Module):
    """Create a target network from the model"""
    target = copy.deepcopy(model)
    for param in target.parameters():
        param.requires_grad = False
    return target

class StockCritic(nn.Module):
    """
    Critic network for an actor-critic (DDPG) model.
    This network estimates Q(s,a) given a state and an action.
    It internally builds the network architecture (using a predictor and two branches)
    and also maintains a target network for stable training.
    """
    def __init__(self, state_dim, action_dim, learning_rate, tau,
                 predictor_type, use_batch_norm):
        """
        Args:
            state_dim (list): Dimensions of the state (e.g., [batch_size, window_length, features] or similar).
            action_dim (list): Dimensions of the action (e.g., [batch_size, nb_actions]).
            learning_rate (float): Learning rate for the critic optimizer.
            tau (float): Soft update parameter for the target network.
            predictor_type (str): Either 'cnn' or 'lstm' to choose the state predictor.
            use_batch_norm (bool): Whether to use batch normalization.
        """
        super(StockCritic, self).__init__()
        self.tau = tau
        self.s_dim = state_dim #(num_stocks, window_length)
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.predictor_type = predictor_type
        self.use_batch_norm = use_batch_norm

        self.q_value = None

        if predictor_type == 'cnn':
            self.predictor = CNNPredictor(input_dim=(1, state_dim[1], state_dim[3]), output_dim=(1, 1), use_batch_norm=use_batch_norm)
        elif predictor_type == 'lstm':
            self.predictor = LSTMPredictor(input_dim=state_dim, output_dim=(1, 1), hidden_dim=64, use_batch_norm=use_batch_norm)
        else:
            raise ValueError('Predictor type not recognized')

        self.fc1_state = nn.Linear(self.s_dim[0] * 64, 64) # num_stocks * 64
        self.fc2_action = nn.Linear(self.a_dim[0], 64)

        # Optional batch normalization after combining the two branches.
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(64)

        # Final layer to produce a single Q-value.
        self.out = nn.Linear(64, 1)
        # Initialize the final layer weights to Uniform[-3e-3, 3e-3]
        nn.init.uniform_(self.out.weight, -0.003, 0.003)
        nn.init.uniform_(self.out.bias, -0.003, 0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.target_network = create_target_network(self)


    def forward(self, state, action):
        """
        Forward pass through the online Q-network.
        
        Args:
            state (torch.Tensor): The state input.
            action (torch.Tensor): The action input.
        
        Returns:
            torch.Tensor: Predicted Q-value.
        """

        state_features = self.predictor(state)  # Expected shape: (batch, 64)
        x_state = self.fc1_state(state_features)
        x_action = self.fc2_action(action)
        x = x_state + x_action
        if self.use_batch_norm:
            x = self.bn(x)
        x = F.relu(x)
        self.q_value = self.out(x)
        return self.q_value

    def train_step(self, state, action, target_q_value):
        """
        Perform a training step on the online network.
        
        Args:
            state (torch.Tensor): The state input.
            action (torch.Tensor): The action taken.
            target_q_value (torch.Tensor): The target Q-value.
        
        Returns:
            tuple: (predicted Q-values, loss value)
        """

        q_value = self.forward(state, action)

        self.optimizer.zero_grad()
        target_q_value = target_q_value.clone().detach() # Ensures that the no_grad context is not entered.
        loss = F.mse_loss(q_value, target_q_value)
        loss.backward()
        self.optimizer.step()
        return q_value, loss.item()

    def predict(self, state, action):
        """
        Predict Q-values using the online network.
        
        Args:
            state (torch.Tensor): The state input.
            action (torch.Tensor): The action input.
        
        Returns:
            torch.Tensor: Predicted Q-values.
        """
        self.eval()
        with torch.inference_mode():
            q_value = self.forward(state, action)
        self.train()
        return q_value

    def predict_target(self, state, action):
        """
        Predict Q-values using the target network.
        
        Args:
            state (torch.Tensor): The state input.
            action (torch.Tensor): The action input.
        
        Returns:
            torch.Tensor: Predicted Q-values from the target network.
        """
        self.target_network.eval()
        with torch.inference_mode():
            q_value = self.target_network(state, action)
        self.target_network.train()
        return q_value

    def action_gradients(self, state, action):
        """
        Compute the gradients of the Q-value with respect to the actions.
        These gradients can be used to update the actor network.
        
        Args:
            state (torch.Tensor): The state input.
            action (torch.Tensor): The action input (should require gradients).
        
        Returns:
            torch.Tensor: Gradients of Q w.r.t. the action.
        """
        # Ensure state does not require gradients.
        state.requires_grad = False
        # Ensure action requires gradients.
        action.requires_grad = True
        q_value = self.forward(state, action)
        q_sum = q_value.sum()
        grad = torch.autograd.grad(q_sum, action, create_graph=True)[0]
        return grad

    def update_target_network(self):
        """
        Soft-update the target network parameters:
            θ_target = τ * θ_online + (1 - τ) * θ_target
        """
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


if __name__ == '__main__':
    critic = StockCritic(state_dim=(10, 20), action_dim=(10), learning_rate=1e-3, tau=1e-3, predictor_type='lstm', use_batch_norm=True)