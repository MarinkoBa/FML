import torch
import torch.nn as nn
import numpy as np


class Q_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(45, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device='cuda:0')
        x = x.reshape(x.shape[0], 45)

        # Takes in a state vector with length state_len and outputs an action vector with length action_len
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)

        return x