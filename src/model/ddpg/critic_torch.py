import torch
import torch.nn as nn
import torch.nn.functional as F
from util import stockPredictor

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, learning_rate, tau, predictor_type, use_batch_norm):
        super(CriticNetwork, self).__init__()
        
        assert isinstance(state_dim, list), 'state_dim must be a list.'
        assert isinstance(action_dim, list), 'action_dim must be a list.'
        
        self.s_dim = state_dim  # [num_stocks, window_length]
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        
        # Create main network
        self.predictor = create_predictor(state_dim, predictor_type)
        
        # Calculate predictor output size
        if predictor_type == 'cnn':
            predictor_output = state_dim[0] * 32  # num_stocks * num_filters
        else:  # lstm
            predictor_output = state_dim[0] * 32  # num_stocks * hidden_dim
        
        # State pathway (after predictor)
        self.state_layer = nn.Linear(predictor_output, 64)
        
        # Action pathway
        self.action_layer = nn.Linear(sum(action_dim), 64)
        
        # Batch norm for combined layer if specified
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn_combined = nn.BatchNorm1d(64)
        
        # Output layer with small initialization
        self.output_layer = nn.Linear(64, 1)
        self.output_layer.weight.data.uniform_(-0.003, 0.003)
        self.output_layer.bias.data.uniform_(-0.003, 0.003)
        
        # Create target network as a separate instance
        self.target_network = None  # Will be initialized after main network is fully set up
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Initialize target network after all components are set up
        self._initialize_target_network()

    def _initialize_target_network(self):
        """Initialize target network as a separate instance with same architecture"""
        self.target_network = CriticNetwork(
            self.s_dim,
            self.a_dim,
            self.learning_rate,
            self.tau,
            self.predictor.predictor_type,
            self.use_batch_norm
        )
        
        # Copy weights from main network
        self.target_network.load_state_dict(self.state_dict())
        
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False

    def forward(self, inputs, action):
        # Process state through predictor
        net = self.predictor(inputs)  # Input shape: [batch, num_stocks, window_length, 1]
        
        # State pathway
        net = self.state_layer(net)
        
        # Action pathway
        action_net = self.action_layer(action)
        
        # Combine pathways
        net = net + action_net
        
        # Apply batch norm if specified
        if self.use_batch_norm:
            net = self.bn_combined(net)
        
        # Final activation and output
        net = F.relu(net)
        return self.output_layer(net)

    def train(self, inputs, action, predicted_q_value):
        """Train the critic network"""
        self.optimizer.zero_grad()
        
        current_q = self.forward(inputs, action)
        loss = self.loss_fn(current_q, predicted_q_value)
        
        loss.backward()
        self.optimizer.step()
        
        return current_q.detach(), loss.item()

    def predict(self, inputs, action):
        """Get Q value prediction"""
        with torch.no_grad():
            return self.forward(inputs, action)

    def predict_target(self, inputs, action):
        """Get target network Q value prediction"""
        with torch.no_grad():
            return self.target_network.forward(inputs, action)

    def action_gradients(self, inputs, action):
        """Get gradients of Q value with respect to actions"""
        action.requires_grad_(True)
        q_value = self.forward(inputs, action)
        q_value.backward(torch.ones_like(q_value))
        gradients = action.grad.data.clone()
        action.requires_grad_(False)
        return gradients

    def update_target_network(self):
        """Update target network parameters using soft update rule"""
        for target_param, param in zip(self.target_network.parameters(), self.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )