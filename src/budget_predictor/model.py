import torch
import torch.nn as nn

class BudgetPredictor(nn.Module):#defining a neural network model
    def __init__(self, input_dim=8, hidden_dim=32, num_classes=4): #input vector is 8 and the nn created a hidden rep of size 32 to learn patterns
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), #first layer , that outputs 32 num
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes) #secomd layer, that outputs 4 values
        )

    def forward(self, x):
        return self.model(x)
