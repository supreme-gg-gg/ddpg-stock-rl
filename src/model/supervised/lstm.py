"""
Train a supervised CNN model using optimal stock as label
"""
import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
from ..base import BaseModel

from torch.utils.data import DataLoader, TensorDataset

from utils.data import normalize


class StockLSTM(nn.Module):
    def __init__(self, num_classes, window_length, weights_file='weights/lstm.h5'):
        self.weights_file = weights_file
        self.num_classes = num_classes
        self.window_length = window_length

        self.lstm = nn.LSTM(input_size=1, hidden_size=20, batch_first=True)
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.num_classes)
        self.dropout = nn.Dropout(0.5)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.graph = torch.get_default_graph()

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.Softmax(self.fc3(x))
        return x

    def save_model(self, path):
        """Save the model state to a file

        Args:
            path (str): Path where to save the model state
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'num_classes': self.num_classes,
            'window_length': self.window_length
        }, path)
        print(f'Model saved successfully to {path}')

    def load_model(self, path):
        """Load the model state from a file

        Args:
            path (str): Path to the saved model state

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.num_classes = checkpoint['num_classes']
            self.window_length = checkpoint['window_length']
            self.eval()  # Set the model to evaluation mode
            print(f'Model loaded successfully from {path}')
            return True
        except Exception as e:
            print(f'Error loading model: {str(e)}')
            return False
        
    def train_step(self, X, Y, criterion):
        self.optimizer.zero_grad()
        


    # def train(self, X_train, Y_train, X_val, Y_val, verbose=True):
    #     continue_train = True
    #     while continue_train:
    #         self.fit(X_train, Y_train, batch_size=64, epochs=50, validation_data=(X_val, Y_val),
    #                        shuffle=True, verbose=verbose)
    #         save_weights = input('Type True to save weights\n')
    #         if save_weights:
    #             self.model.save(self.weights_file)
    #         continue_train = input("True to continue train, otherwise stop training...\n")
    #     print('Finish.')

    def train(self, X_train, Y_train, X_val, Y_val, verbose=True):
        continue_train = True
        # Convert numpy arrays to tensors once (or in each epoch if your data is changing)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        X_val_tensor   = torch.tensor(X_val, dtype=torch.float32)
        Y_val_tensor   = torch.tensor(Y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset   = TensorDataset(X_val_tensor, Y_val_tensor)
        
        while continue_train:
            # Create dataloaders; shuffling training data.
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # Train for a fixed number of epochs (50 here)
            for epoch in range(100):
                self.train()
                running_loss = 0.0
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    predictions = self.forward(x_batch)
                    loss = nn.CrossEntropyLoss(predictions, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    running_loss += loss.item() * x_batch.size(0)
                
                epoch_loss = running_loss / len(train_dataset)
                if verbose:
                    print(f"Epoch {epoch+1}/50, Training Loss: {epoch_loss:.4f}")

    def evaluate(self, X_test, Y_test, verbose=False):
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        self.eval()
        acc = 0
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                predictions = self.model(x_val)
                loss = nn.CrossEntropyLoss(predictions, y_val)
                acc += (predictions.argmax(dim=1) == y_val).float().mean()
                val_loss += loss.item() * x_val.size(0)
        avg_val_loss = val_loss / len(test_dataset)
        acc = acc / len(test_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        return acc, avg_val_loss

    def predict(self, X_test, verbose=False):
        return self.forward(X_test, verbose=verbose)

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        action = np.zeros((self.num_classes,))
        obsX = observation[:, -self.window_length:, 3] / observation[:, -self.window_length:, 0]
        obsX = normalize(obsX)
        obsX = np.expand_dims(obsX, axis=0)
        
        obsX_tensor = torch.tensor(obsX, dtype=torch.float32, device=self.device)
        

        self.eval()
        with torch.no_grad():
            outputs = self.forward(obsX_tensor)
            _, predicted = torch.max(outputs, dim=1)
            current_action_index = predicted.item()
        
        # Set the corresponding action to 1.0
        action[current_action_index] = 1.0
        return action
