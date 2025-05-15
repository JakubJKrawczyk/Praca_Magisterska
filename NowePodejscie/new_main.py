import os
import json
import datetime
from os.path import isfile, join

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss

from Custom.Data.DataHelper import DataHelper
from EEG_classificator import ImprovedEEGClassifier  # zakładamy, że model jest w pliku `model.py`
from DataHelper import DataHelper, EmotionDataset

# ========================== SEKCJA 1: ŁADOWANIE DANYCH ==========================
print("🔹 [1/5] Ładowanie danych...")

# Dummy data for example: [batch, time_samples, freq_bands, probes]
data_dir = "EEG_data"
data_loaded = EmotionDataset([])
seed = 42
data_files = [f for f in os.listdir(data_dir) if isfile(join(data_dir, f))]

for file_idx, file in enumerate(data_files):
    path = os.path.join(data_dir, file)

    data = DataHelper.load_mat_file(path)

    processed_data = DataHelper.adapt_to_emotion_format(data)
    data_loaded.extend(processed_data)

emotion_dataset = data_loaded.tensorize()


# ====================== SEKCJA 2: PRZYGOTOWANIE DANYCH ==========================
print("🔹 [2/5] Przygotowanie danych...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"📦 Używane urządzenie: {device}")

dataset_len = len(data_loaded)
train_split = int(0.8 * dataset_len)
val_split = dataset_len - train_split
generator1 = torch.Generator().manual_seed(seed)
generator2 = torch.Generator().manual_seed(seed)

eeg_values = emotion_dataset.tensors[0]
emotion_ids = emotion_dataset.tensors[1]

train_dataset, val_dataset = torch.utils.data.random_split(emotion_dataset, [train_split, val_split])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ================= SEKCJA 3: PRZYGOTOWANIE TRENINGU =============================
print("🔹 [3/5] Konfiguracja modelu i treningu...")

# Parametry
params = {
    "input_time_samples": 128,
    "num_frequency_bands": 5,
    "num_probes": 62,
    "num_classes": 2,
    "temporal_filters": 64,
    "spatial_filters": 128,
    "transformer_dim": 256,
    "nhead": 8,
    "num_layers": 4,
    "dropout": 0.3,
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 0.001
}

model = ImprovedEEGClassifier(
    input_time_samples=params["input_time_samples"],
    num_frequency_bands=params["num_frequency_bands"],
    num_probes=params["num_probes"],
    num_classes=params["num_classes"],
    temporal_filters=params["temporal_filters"],
    spatial_filters=params["spatial_filters"],
    transformer_dim=params["transformer_dim"],
    nhead=params["nhead"],
    num_layers=params["num_layers"],
    dropout=params["dropout"]
).to(device)

criterion = CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=params["learning_rate"])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"])

# ========================== SEKCJA 4: TRENING ===================================
print("🔹 [4/5] Start treningu...")

train_history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

for epoch in range(params["epochs"]):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    avg_train_loss = running_loss / len(train_loader)
    train_history["train_loss"].append(avg_train_loss)

    # Ewaluacja
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            outputs = model(X_val_batch)
            loss = criterion(outputs, y_val_batch)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_val_batch).sum().item()
            total += y_val_batch.size(0)

    avg_val_loss = val_loss / len(val_loader)
    accuracy = correct / total
    train_history["val_loss"].append(avg_val_loss)
    train_history["val_accuracy"].append(accuracy)

    print(f"📘 Epoka {epoch+1}/{params['epochs']} | "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {accuracy:.4f}")

# ========================== SEKCJA 5: ZAPIS WYNIKÓW =============================
print("🔹 [5/5] Zapis modelu i wyników...")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"trained_models/run_{timestamp}"
os.makedirs(save_dir, exist_ok=True)

torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

with open(os.path.join(save_dir, "params.json"), "w") as f:
    json.dump(params, f, indent=2)

with open(os.path.join(save_dir, "training_log.json"), "w") as f:
    json.dump(train_history, f, indent=2)

print(f"✅ Model zapisany do folderu: {save_dir}")
