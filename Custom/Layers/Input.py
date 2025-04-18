from torch import nn, torch
import math

def expand_value_sinusoidal(input_tensor: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Rozwija tensor do odpowiedniego kształtu za pomocą sinusoidalnego kodowania.
    Obsługuje zarówno pojedyncze przykłady (x, 32) jak i batche (batch_size, x, 32).
    """
    if input_tensor is None:
        raise Exception("Input tensor is NONE")

    # Sprawdzamy wymiary tensora wejściowego
    print(f"Input tensor shape in expand_value_sinusoidal: {input_tensor.shape}")
    
    # Dla danych w kształcie [batch_size, features]
    if len(input_tensor.shape) == 2:
        batch_size, features = input_tensor.shape
        
        # Przygotuj tensor wyjściowy
        output = torch.zeros(batch_size, d_model, dtype=torch.float32, device=input_tensor.device)
        
        # Dla każdej próbki w batchu
        for i in range(batch_size):
            # Pobierz wektor cech dla jednej próbki
            sample = input_tensor[i]  # [features]
            
            # Rozszerz do wymiaru [features, 1]
            position = sample.unsqueeze(-1)
            
            # Oblicz współczynniki dla kodowania sinusoidalnego
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32, device=input_tensor.device) * 
                -(math.log(10000.0) / d_model)
            )
            
            # Oblicz kodowanie sinusoidalne
            sinusoid_input = position * div_term  # [features, d_model/2]
            sin_values = torch.sin(sinusoid_input)
            cos_values = torch.cos(sinusoid_input)
            
            # Uśrednij wartości po wszystkich cechach
            sin_mean = sin_values.mean(dim=0)  # [d_model/2]
            cos_mean = cos_values.mean(dim=0)  # [d_model/2]
            
            # Przepleć wartości sin i cos
            output[i, 0::2] = sin_mean
            output[i, 1::2] = cos_mean
            
        return output
    else:
        raise ValueError(f"Nieobsługiwany kształt tensora wejściowego: {input_tensor.shape}")


class InputLayer(nn.Module):
    def __init__(self, expected_vector_size):
        super(InputLayer, self).__init__()
        self.size = expected_vector_size
        self.expand_values = expand_value_sinusoidal
        
    def forward(self, x):
        # Print shape for debugging
        print(f"Input shape: {x.shape}")
        
        # Przetwarzamy dane wejściowe
        result = self.expand_values(x, self.size)
        
        # Print result shape for debugging
        print(f"Output shape: {result.shape}")
        
        return result
