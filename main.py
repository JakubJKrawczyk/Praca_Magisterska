from operator import index
from Custom.Data.DataHelper import DataHelper, EmotionDataset
from os import listdir
from os.path import isfile, join
from Custom.Data.Logger import Logger
from Custom.Models.EEG_clas_model import EEG_class_model
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

data_path = "Custom/Data/EEG_data/"
data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
logger = Logger()

# Load data
data_per_user = EmotionDataset([])
for file in data_files:
    path = join(data_path, file)
    data = DataHelper.load_mat_file(path, True)
    processed_data = DataHelper.prepare_data(data)
    data_per_user.extend(processed_data)
    logger.display_progress(data_files.index(file), len(data_files), "Processing data", f"Processed {file}")

print("Utworzony Tensor Dataset")
tensor_dataset = data_per_user.tensorize()  # Zwraca TensorDataset
print(tensor_dataset.tensors)

# Rozdziel dane na emocje (etykiety) i wartości EEG (cechy)
emotion_ids = tensor_dataset.tensors[0]  # Pierwszy tensor to ID emocji
eeg_values = tensor_dataset.tensors[1]   # Drugi tensor to wartości EEG

print(f"Liczba wszystkich próbek: {len(emotion_ids)}")

# Podział na zbiory treningowe i testowe w proporcji 9:1
train_size = int(0.9 * len(emotion_ids))
test_size = len(emotion_ids) - train_size

# Podział dataset'u na treningowy i testowy
train_dataset, test_dataset = random_split(
    tensor_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # Dla powtarzalności wyników
)

print(f"Liczba próbek treningowych: {len(train_dataset)}")
print(f"Liczba próbek testowych: {len(test_dataset)}")

# Utworzenie DataLoader dla łatwiejszego przetwarzania podczas treningu
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\n\nSekcja modelu")
print("Tworzenie obiektu modelu")
model = EEG_class_model(d_model=len(data_per_user.values))
print("Model stworzony........")