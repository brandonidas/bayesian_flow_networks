import torch  # The top-level PyTorch package and tensor library.
import torch.nn as nn 

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.fc(x)
    
'''
Epoch [9900/10000], Loss: 0.1049
Epoch [10000/10000], Loss: 0.1074
Test Loss: 0.1146
'''