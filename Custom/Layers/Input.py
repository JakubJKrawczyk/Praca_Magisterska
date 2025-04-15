import jax.numpy as jnp
from pyarrow import Tensor
from torch import nn, torch

def expand_value_sinusoidal(input_tensor, vector_size) -> Tensor:
    """
    Metoda 1: Kodowanie sinusoidalne - podobne do kodowania pozycyjnego w transformerach.
    Przekształca pojedynczą wartość w wektor za pomocą funkcji sinusoidalnych o różnej częstotliwości.
    """
    # Dane wejściowe w formacie (x, 32)
    if input_tensor is None:
        raise Exception("Input tensor is NONE")

    result = torch.zeros((input_tensor.size(0),input_tensor.size(1), vector_size), dtype=jnp.float32)
    freqs = 10 ** (jnp.arange(0, vector_size // 2) / (vector_size // 2 - 1) * 4.0)  # Częstotliwości od 1 do 10000
    for sample in range(input_tensor.size(0)):
        for node in range(input_tensor.size(1)):
            for expected_value in range(vector_size):
                result[sample][node][expected_value] = torch.tensor(jnp.sin(input_tensor[sample][node] * freqs[expected_value]))

    return result


class InputLayer(nn.Module):
    def __init__(self, expected_vector_size):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.expand_values = expand_value_sinusoidal
    def forward(self, x):
        result = self.expand_values(x, self.size)
        print("InputLayer forward")
        if result.ndim[2].size == self.size:
            return result
        else:
            yield Exception("Invalid shape")
