import torch.nn as nn

from config import device


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model, device=device)
        self.relu = nn.ReLU()
        self.layerNorm = nn.LayerNorm(d_model, device=device)
    def forward(self, x):
        start = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.layerNorm(x + start)
        return x
