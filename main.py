from operator import index
from Custom.Data.DataHelper import DataHelper, EmotionDataset
from os import listdir, makedirs
from os.path import isfile, join, exists
from Custom.Data.Logger import Logger
from Custom.Models.EEG_clas_model import EEG_class_model
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from Trainer import Trainer

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

# Konfiguracja treningu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")
model = model.to(device)

# Kryterium oceny - dla klasyfikacji używamy Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

# Definicja parametrów optymalizatora ADAM
learning_rate = 0.001        # Szybkość uczenia
beta1 = 0.9                  # Współczynnik beta1 dla średnich pierwszego rzędu
beta2 = 0.999                # Współczynnik beta2 dla średnich drugiego rzędu
epsilon = 1e-08              # Wartość epsilon dla stabilności numerycznej
weight_decay = 0.0001        # Regularyzacja L2 (decay wag)

# Optymalizator ADAM z parametrami - przekazujemy wszystkie parametry modelu
optimizer = optim.Adam(
    model.parameters(),      # Wszystkie parametry modelu
    lr=learning_rate,        # Współczynnik uczenia
    betas=(beta1, beta2),    # Współczynniki dla średnich wykładniczych
    eps=epsilon,             # Stała dla stabilności numerycznej
    weight_decay=weight_decay # Regularyzacja L2
)

# Trenowanie modelu
print("Rozpoczęcie treningu...")
trainer = Trainer(device)

num_epochs = 15
train_losses, train_accuracies = trainer.train_model(model, train_loader, criterion, optimizer, num_epochs)

# Ewaluacja modelu
print("Ewaluacja modelu...")
test_loss, test_accuracy = trainer.evaluate_model(model, test_loader, criterion)

# Zapisanie modelu z nazwą zawierającą datę
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = f"models/trained-{current_date}"

# Upewnij się, że folder istnieje
if not exists(save_dir):
    makedirs(save_dir)

# Zapisz model
model_path = join(save_dir, "eeg_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'epochs': num_epochs,
}, model_path)

print(f"Model zapisany w: {model_path}")

# Wizualizacja wyniku treningu
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Strata podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Strata')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title('Dokładność podczas treningu')
plt.xlabel('Epoka')
plt.ylabel('Dokładność (%)')

plt.tight_layout()
plt.savefig(join(save_dir, "training_metrics.png"))
plt.show()

print(f"Końcowa dokładność na zbiorze testowym: {test_accuracy:.2f}%")