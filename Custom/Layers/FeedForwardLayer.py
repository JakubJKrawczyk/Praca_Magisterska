import torch.nn as nn

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.layerNorm = nn.LayerNorm(d_model)
    def forward(self, x):
        start = x
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.layerNorm(x + start)
        return x
