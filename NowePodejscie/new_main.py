import os
import json
import datetime
import random
import numpy as np
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from EEG_classificator import AdaptiveEEGClassifier
from DataHelper import DataHelper

# ========================== SEKCJA 1: ŁADOWANIE DANYCH ==========================
print("🔹 [1/5] Ładowanie danych...")

data_dir = "EEG_data"
data_files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]
seed = 42

# Ustawienie ziarna dla powtarzalnych wyników
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Tworzenie walidacji krzyżowej z filmami
all_folds = []

for file_idx, file in enumerate(data_files):
    path = os.path.join(data_dir, file)
    print(f"Przetwarzanie pliku {file_idx + 1}/{len(data_files)}: {file}")

    # Tworzenie foldów walidacji krzyżowej z 5 oknami dla najkrótszego filmu
    cv_folds = DataHelper.create_cv_datasets_per_film(
        path,
        num_folds=5,
        num_windows=5,  # Najkrótszy film będzie miał 5 okien
        overlap=0.5  # 50% nakładanie się okien
    )
    all_folds.extend(cv_folds)

# ====================== SEKCJA 2: PRZYGOTOWANIE DANYCH ==========================
print("🔹 [2/5] Przygotowanie danych...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Używane urządzenie: {device}")

# Używamy losowego foldu do treningu i walidacji
fold_idx = 0  # Można zmienić lub iterować po wszystkich
train_loader, val_loader = all_folds[fold_idx]

# Sprawdzenie rozmiaru okna i innych parametrów
# Poprawka: Prawidłowe wyciąganie danych z batcha
batch = next(iter(train_loader))
window_batch = batch[0]  # Tensor z danymi okien
label_batch = batch[1]  # Tensor z etykietami

# Wydruk dla weryfikacji struktury
print(f"Batch type: {type(batch)}")
print(f"Window batch shape: {window_batch.shape}")
print(f"Label batch shape: {label_batch.shape}")

# Uzyskanie wymiarów dla konfiguracji modelu
window_size = window_batch.shape[2]  # [num_windows, bands, time, probes]
num_bands = window_batch.shape[3]
num_probes = window_batch.shape[4]

print(f"Rozmiar okna: {window_size} próbek")
print(f"Liczba pasm: {num_bands}")
print(f"Liczba elektrod: {num_probes}")

# ================= SEKCJA 3: PRZYGOTOWANIE TRENINGU =============================
print("🔹 [3/5] Konfiguracja modelu i treningu...")

# Parametry
params = {
    "window_size": window_size,
    "window_stride": window_size // 2,  # 50% nakładanie
    "num_frequency_bands": num_bands,
    "num_probes": num_probes,
    "num_classes": 7,  # 7 emocji
    "temporal_filters": 64,
    "spatial_filters": 128,
    "transformer_dim": 256,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.35,
    "epochs": 200,
    "batch_size": 1,  # 1 film na raz
    "learning_rate": 0.001
}

# Inicjalizacja modelu
# In your main.py
model = AdaptiveEEGClassifier(
    window_size=params["window_size"],  # Match your data's time dimension
    num_frequency_bands=params["num_frequency_bands"],  # Match your data's band dimension
    num_probes=params["num_probes"],  # Match your data's probe dimension
    num_classes=params["num_classes"],  # Number of classes
    dropout=params["dropout"]
).to(device)

# Wypisanie liczby parametrów modelu
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Liczba parametrów modelu: {num_params:,}")

# Definiowanie optymalizatora, funkcji straty i schedulera
criterion = CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

# ========================== SEKCJA 4: TRENING ===================================
print("🔹 [4/5] Start treningu...")

train_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

for epoch in range(params["epochs"]):
    # Tryb treningowy
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        # Poprawka: Prawidłowe wyciąganie danych z batcha
        film_windows = batch[0].to(device)  # [num_windows, bands, time, probes]
        film_label = batch[1].to(device)  # [1]

        # Przetwarzanie każdego okna z filmu
        window_outputs = []
        optimizer.zero_grad()

        for window_idx in range(film_windows.shape[0]):
            window = film_windows[window_idx].unsqueeze(0)  # [1, bands, time, probes]
            output = model(window)  # [1, num_classes]
            window_outputs.append(output)

        # Średnia predykcja ze wszystkich okien
        film_output = torch.mean(torch.stack(window_outputs), dim=0)  # [1, num_classes]

        # Obliczanie straty
        loss = criterion(film_output, film_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Aktualizacja schedulera
    scheduler.step()
    avg_train_loss = running_loss / len(train_loader)
    train_history["train_loss"].append(avg_train_loss)

    # Ewaluacja
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            # Poprawka: Prawidłowe wyciąganie danych z batcha
            film_windows = batch[0].to(device)  # [num_windows, bands, time, probes]
            film_label = batch[1].to(device)  # [1]

            # Przetwarzanie każdego okna z filmu
            window_outputs = []

            for window_idx in range(film_windows.shape[0]):
                window = film_windows[window_idx].unsqueeze(0)  # [1, bands, time, probes]
                output = model(window)  # [1, num_classes]
                window_outputs.append(output)

            # Średnia predykcja ze wszystkich okien
            film_output = torch.mean(torch.stack(window_outputs), dim=0)  # [1, num_classes]

            # Obliczanie straty
            loss = criterion(film_output, film_label)
            val_loss += loss.item()

            # Obliczanie dokładności
            preds = torch.argmax(film_output, dim=1)
            correct += (preds == film_label).sum().item()
            total += film_label.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    train_history["val_loss"].append(avg_val_loss)
    train_history["val_accuracy"].append(accuracy)

    print(f"📘 Epoka {epoch + 1}/{params['epochs']} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {accuracy:.4f}")

# ========================== SEKCJA 5: ZAPIS WYNIKÓW =============================
print("🔹 [5/5] Zapis modelu i wyników...")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"trained_models/run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

# Zapis modelu
torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

# Zapis parametrów
with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f, indent=2)

# Zapis historii treningu
with open(os.path.join(save_dir, "training_log.json"), "w") as f:
    json.dump(train_history, f, indent=2)

# Zapis podsumowania
with open(os.path.join(save_dir, "summary.txt"), "w") as f:
    f.write(f"Model trenowany: {timestamp}\n")
    f.write(f"Urządzenie: {device}\n")
    f.write(f"Liczba parametrów: {num_params:,}\n")
    f.write(f"Rozmiar okna: {window_size}\n")
    f.write(f"Najlepsza dokładność walidacji: {max(train_history['val_accuracy']):.4f}\n")
    f.write(f"Końcowa dokładność walidacji: {train_history['val_accuracy'][-1]:.4f}\n")

print(f"✅ Model zapisany do folderu: {save_dir}")
print(f"✅ Najlepsza dokładność walidacji: {max(train_history['val_accuracy']):.4f}")