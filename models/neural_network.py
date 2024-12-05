"""
Resources
---------
https://stackoverflow.com/a/68609343/8728749
HW 6
"""

import torch.nn.functional as F
import torch.nn as nn


class RegressionNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=36, out_features=12)
        self.fc2 = nn.Linear(in_features=12, out_features=1)
        # research on good structures to use here

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
