from torch import nn, torch
import math

def expand_value_sinusoidal(input_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Rozwija tensor do odpowiedniego kształtu za pomocą sinusoidalnego kodowania.
    Obsługuje zarówno pojedyncze przykłady (x, 32) jak i batche (batch_size, x, 32).
    """
    if input_tensor is None:
        raise Exception("Input tensor is NONE")

    # Sprawdzamy czy mamy batch czy pojedynczy przykład
    is_batch = input_tensor.dim() == 3
    
    if not is_batch:
        # Pojedynczy przykład (seq_len, features)
        num_sensors = input_tensor.shape[0]  # seq_len
        
        # Częstotliwości dla sinusoid
        position = input_tensor.unsqueeze(-1)  # (seq_len, features, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
        ).to(input_tensor.device)  # (d_model/2)
        
        # Obliczamy sinusoidalne kodowanie
        sinusoid_input = position * div_term  # (seq_len, features, d_model/2)
        sin_embed = torch.sin(sinusoid_input)
        cos_embed = torch.cos(sinusoid_input)
        
        # Tworzymy pusty tensor i przeplatamy sinusoidy
        output = torch.zeros(num_sensors, d_model, dtype=torch.float32).to(input_tensor.device)
        output[..., 0::2] = sin_embed
        output[..., 1::2] = cos_embed
        
        return output
    else:
        # Batch (batch_size, seq_len, features)
        batch_size = input_tensor.shape[0]
        seq_len = input_tensor.shape[1]  # seq_len
        
        # Przetwarzamy każdy przykład w batchu osobno
        outputs = []
        for i in range(batch_size):
            single_input = input_tensor[i]  # (seq_len, features)
            
            # Częstotliwości dla sinusoid
            position = single_input.unsqueeze(-1)  # (seq_len, features, 1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model)
            ).to(input_tensor.device)  # (d_model/2)
            
            # Obliczamy sinusoidalne kodowanie
            sinusoid_input = position * div_term  # (seq_len, features, d_model/2)
            sin_embed = torch.sin(sinusoid_input)
            cos_embed = torch.cos(sinusoid_input)
            
            # Tworzymy pusty tensor i przeplatamy sinusoidy
            output = torch.zeros(seq_len, d_model, dtype=torch.float32).to(input_tensor.device)
            output[..., 0::2] = sin_embed
            output[..., 1::2] = cos_embed
            
            outputs.append(output)
        
        # Łączymy wyniki dla całego batcha
        return torch.stack(outputs)  # (batch_size, seq_len, d_model)


class InputLayer(nn.Module):
    def __init__(self, expected_vector_size):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.expand_values = expand_value_sinusoidal
    def forward(self, x):
        # Print shape for debugging
        print(f"Input shape: {x.shape}")
        
        result = self.expand_values(x, self.size)
        
        # Print result shape for debugging
        print(f"Output shape: {result.shape}")
        
        # Sprawdzamy kształt wyniku w zależności od tego czy mamy batch czy pojedynczy przykład
        if x.dim() == 2:  # Pojedynczy przykład
            if result.shape[1] == self.size:
                return result
            else:
                raise Exception(f"Invalid shape: expected second dimension to be {self.size}, got {result.shape}")
        else:  # Batch
            if result.shape[2] == self.size:
                return result
            else:
                raise Exception(f"Invalid shape: expected third dimension to be {self.size}, got {result.shape}")
