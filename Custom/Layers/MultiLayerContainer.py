

class MultiSequenceContainer:
    def __init__(self, sequences):
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

    def zero_grad(self):
        for layer in self.sequences:
            layer.zero_grad()