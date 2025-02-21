import torch
import torch.nn as nn

def create_predictor_model(
        inputs: torch.Tensor, 
        predictor_type: str, 
        use_batch_norm: bool) -> nn.Sequential:
    """
    Create a predictor model for the StockActor and StockCritic class.
    """
    assert predictor_type in ['cnn', 'lstm'], 'type must be either cnn or lstm'
    num_stock = inputs.size(1)
    window_length = inputs.size(2)
    
    if predictor_type == 'cnn':
        layers = []
        # First conv layer
        layers.append(nn.Conv2d(1, 32, kernel_size=(1, 3)))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        
        # Second conv layer
        layers.append(nn.Conv2d(32, 32, kernel_size=(1, window_length - 2)))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        
        # Flatten layer
        layers.append(nn.Flatten())

        print("Shape after conv layers: ", layers[-1].shape)
        
        return nn.Sequential(*layers)
        
    elif predictor_type == 'lstm':
        hidden_dim = 32

        # reshape to [batch_size * num_stock, window_length, 1]
        inputs = inputs.reshape(-1, window_length, 1)
        
        return nn.Sequential(
            nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True),
            # Reshape will need to be handled outside the Sequential
            # nn.Flatten()
        )
    
        # lstm_out = lstm_out.view(batch_size, self.num_stock * self.hidden_dim)
    
    else:
        raise NotImplementedError

def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, num_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observations