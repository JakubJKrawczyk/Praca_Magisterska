import torch.nn as nn

class MultiSequenceContainer(nn.Module):
    def __init__(self, sequences):
        super(MultiSequenceContainer, self).__init__()
        self.sequences = sequences

    def forward(self, x):
        for sequence in self.sequences:
            for layer in sequence:
                x = layer.forward(x)
        return x

    def backward(self, dLdy):
        for layer in reversed(self.sequences):
            dLdy = layer.backward(dLdy)
        return dLdy

    def step(self, learning_rate):
        for layer in self.sequences:
            layer.step(learning_rate)

    def zero_grad(self, **kwargs):
        for layer in self.sequences:
            layer.zero_grad()