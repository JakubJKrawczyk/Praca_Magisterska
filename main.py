from operator import index
from time import sleep

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
from config import device

# Konfiguracja ogólna
data_path = "Custom/Data/EEG_data/"
data_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
logger = Logger()

prepare_process_steps = 9

# Adjust these parameters
d_model = 128        # Wymiar modelu
num_heads = 4        # Liczba głowic uwagi
num_layers = 2       # Liczba warstw enkodera
dropout = 0.3        # Współczynnik dropout (increased for better regularization)
num_classes = 7      # Liczba klas emocji

# Parametry treningu
batch_size = 64      # Rozmiar partii danych (increased for better stability)
num_epochs = 50      # Liczba epok treningu
learning_rate = 0.0005  # Szybkość uczenia (reduced for better convergence)
beta1 = 0.9          # Współczynnik beta1 dla średnich pierwszego rzędu
beta2 = 0.999        # Współczynnik beta2 dla średnich drugiego rzędu
epsilon = 1e-08      # Wartość epsilon dla stabilności numerycznej
weight_decay = 0.001  # Regularyzacja L2 (increased for better generalization)
# 1. Ładowanie i przetwarzanie danych
logger.display_progress(0, prepare_process_steps, "Starting data processing", "Initializing...")

# Ładowanie danych
data_per_user = EmotionDataset([])
for file in data_files:
    path = join(data_path, file)
    data = DataHelper.load_mat_file(path, True)
    processed_data = DataHelper.adapt_fft_to_original_format(data)
    data_per_user.extend(processed_data)
    logger.display_progress(data_files.index(file), len(data_files), "Processing data", f"Processed {file}")

# exit(0)
# Konwersja do tensor dataset
logger.display_progress(1, prepare_process_steps, "Preparing for training", f"Creating tensor dataset...")
tensor_dataset = data_per_user.tensorize()  # Zwraca TensorDataset
print(f"Tensor dataset created. Shapes: {[t.shape for t in tensor_dataset.tensors]}")

# Podział na etykiety i wartości EEG
logger.display_progress(2, prepare_process_steps, "Preparing for training", f"Splitting dataset on labels and values...")
emotion_ids = tensor_dataset.tensors[0]  # Pierwszy tensor to ID emocji
eeg_values = tensor_dataset.tensors[1]   # Drugi tensor to wartości EEG
print(f"Total samples: {len(emotion_ids)}")

# Sprawdzenie wymiarów danych
print(f"EEG values shape: {eeg_values.shape}")
print(f"Emotion IDs shape: {emotion_ids.shape}")

# Podział na zbiory treningowe i testowe
logger.display_progress(3, prepare_process_steps, "Preparing for training", f"Splitting dataset on training and testing dataset...")
train_size = int(0.8 * len(emotion_ids))
test_size = len(emotion_ids) - train_size

# Podział z ustalonym seed dla powtarzalności
train_dataset, test_dataset = random_split(
    tensor_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Utworzenie DataLoader z odpowiednim batch_size
logger.display_progress(4, prepare_process_steps, "Preparing for training", f"Preparing dataloaders...")
# Używaj mniejszych batch_size dla lepszego uczenia
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Number of batches in training: {len(train_loader)}")
print(f"Number of batches in testing: {len(test_loader)}")

# 2. Tworzenie modelu
logger.display_progress(5, prepare_process_steps, "Preparing for training", f"Creating model...")
print("\nModel section")
print("Creating model object")

# Tworzenie modelu z poprawionymi parametrami
model = EEG_class_model(
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
    num_classes=num_classes
)

print("Model created successfully")

# Przeniesienie modelu na odpowiednie urządzenie
logger.display_progress(6, prepare_process_steps, "Preparing for training", f"Moving model to {device}...")
model.to(device)
print(f"Using device: {device}")

# 3. Konfiguracja treningu
logger.display_progress(7, prepare_process_steps, "Preparing for training", f"Setting up loss function...")
# Kryterium oceny - dla klasyfikacji używamy Cross Entropy Loss
criterion = nn.CrossEntropyLoss()

logger.display_progress(8, prepare_process_steps, "Preparing for training", f"Creating optimizer...")
# Optymalizator Adam z ustawionymi parametrami
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    eps=epsilon,
    weight_decay=weight_decay
)

# 4. Trening modelu
logger.display_progress(9, prepare_process_steps, "Starting training...", f"Initializing trainer...")
# Inicjalizacja trener z odpowiednimi parametrami
trainer = Trainer(device, log_interval=5, clip_grad=1.0)

print(f"Starting training for {num_epochs} epochs...")
train_losses, train_accuracies = trainer.train_model(model, train_loader, criterion, optimizer, num_epochs, visualize=True)

# 5. Ewaluacja modelu
print("Evaluating model...")
test_loss, test_accuracy = trainer.evaluate_model(model, test_loader, criterion)

# 6. Zapisanie wyników
logger.display_progress(10, prepare_process_steps, "Saving results...", f"Creating directory...")

# Przygotowanie nazwy katalogu z datą i dokładnością
current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
accuracy_str = f"{test_accuracy:.1f}".replace(".", "_")
save_dir = f"models/trained-{current_date}-acc{accuracy_str}"

# Utworzenie katalogu, jeśli nie istnieje
if not exists(save_dir):
    makedirs(save_dir)

# Zapisanie modelu wraz z parametrami i wynikami
model_path = join(save_dir, "eeg_emotion_model.pth")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_loss': test_loss,
    'test_accuracy': test_accuracy,
    'epochs': num_epochs,
    'model_config': {
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': dropout,
        'num_classes': num_classes
    }
}, model_path)

print(f"Model saved to: {model_path}")

# 7. Wizualizacja wyników
plt.figure(figsize=(15, 6))

# Wykres strat
plt.subplot(1, 2, 1)
plt.plot(train_losses, 'b-', label='Training Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Wykres dokładności
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, 'r-', label='Training Accuracy')
plt.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}%')
plt.title('Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(join(save_dir, "training_results.png"))
plt.show()

# Zapisanie szczegółowych informacji o treningu
with open(join(save_dir, "training_details.txt"), "w") as f:
    f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("Model Configuration:\n")
    f.write(f"- d_model: {d_model}\n")
    f.write(f"- num_heads: {num_heads}\n")
    f.write(f"- num_layers: {num_layers}\n")
    f.write(f"- dropout: {dropout}\n")
    f.write(f"- num_classes: {num_classes}\n\n")

    f.write("Training Parameters:\n")
    f.write(f"- batch_size: {batch_size}\n")
    f.write(f"- num_epochs: {num_epochs}\n")
    f.write(f"- learning_rate: {learning_rate}\n")
    f.write(f"- weight_decay: {weight_decay}\n\n")

    f.write("Results:\n")
    f.write(f"- Final training loss: {train_losses[-1]:.6f}\n")
    f.write(f"- Final training accuracy: {train_accuracies[-1]:.2f}%\n")
    f.write(f"- Test loss: {test_loss:.6f}\n")
    f.write(f"- Test accuracy: {test_accuracy:.2f}%\n")

logger.display_progress(11, prepare_process_steps, "Training completed!", f"Final test accuracy: {test_accuracy:.2f}%")
sleep(1)

print(f"Final test accuracy: {test_accuracy:.2f}%")
print(f"All results saved to: {save_dir}")
