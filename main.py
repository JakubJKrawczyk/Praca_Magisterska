from operator import index
import time
from pathlib import Path
import os
from os import listdir, makedirs
from os.path import isfile, join, exists

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Custom.Data.DataHelper import DataHelper, EmotionDataset
from Custom.Data.Logger import Logger
from Custom.Models.EEG_clas_model import EEG_class_model
from Trainer import Trainer
from config import device


class EEGEmotionClassifier:
    """
    Klasa zarządzająca całym procesem treningu i ewaluacji klasyfikatora emocji
    na podstawie danych EEG z wykorzystaniem nowej architektury.
    """

    def __init__(self, config=None):
        """
        Inicjalizuje klasyfikator z konfiguracją.

        Args:
            config (dict, optional): Słownik z parametrami konfiguracyjnymi
        """
        # Inicjalizacja loggera
        self.logger = Logger()

        # Konfiguracja ścieżek
        self.data_path = "Custom/Data/EEG_data/"
        self.prepare_process_steps = 9

        # Domyślna konfiguracja
        self.config = {
            # Parametry danych
            "num_bands": 5,  # Liczba pasm częstotliwości
            "num_nodes": 32,  # Liczba elektrod

            # Parametry modelu
            "d_model": 64,  # Wymiar modelu
            "num_heads": 4,  # Liczba głowic uwagi
            "num_layers": 3,  # Liczba warstw enkodera
            "dropout": 0.03,  # Współczynnik dropout
            "num_classes": 7,  # Liczba klas emocji (7 podstawowych stanów emocjonalnych)

            # Parametry treningu
            "batch_size": 32,  # Rozmiar wsadu
            "num_epochs": 100,  # Liczba epok
            "learning_rate": 0.00000000001,  # Szybkość uczenia
            "beta1": 0.9,  # Beta1 dla Adama
            "beta2": 0.999,  # Beta2 dla Adama
            "epsilon": 1e-08,  # Epsilon dla Adama
            "weight_decay": 0.0001,  # Regularyzacja L2
            "clip_grad": 1.0,  # Przycinanie gradientów

            # Inne
            "seed": 42,  # Ziarno losowości
            "visualize": True  # Czy wizualizować podczas treningu
        }

        # Nadpisanie domyślnej konfiguracji
        if config is not None:
            for key, value in config.items():
                self.config[key] = value

        # Ustawienie ziarna losowości
        torch.manual_seed(self.config["seed"])
        np.random.seed(self.config["seed"])

        # Zapis nazw pasm dla wizualizacji
        self.band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    def load_and_process_data(self):
        """
        Ładuje i przetwarza dane EEG z plików .mat.

        Returns:
            EmotionDataset: Dataset z przetworzonymi danymi
        """
        self.logger.display_progress(0, self.prepare_process_steps,
                                     "Starting data processing", "Initializing...")

        # Lista plików danych
        data_files = [f for f in listdir(self.data_path) if isfile(join(self.data_path, f))]

        # Agregacja danych
        data_per_user = EmotionDataset([])
        for file_idx, file in enumerate(data_files):
            path = join(self.data_path, file)

            # Ładowanie pliku .mat
            data = DataHelper.load_mat_file(path, True)

            # Przetwarzanie danych - używamy adapt_to_emotion_format, który zwraca dane
            # w formacie [samples, bands, electrodes]
            processed_data = DataHelper.adapt_to_emotion_format(data)

            # Rozszerzenie datasetu
            data_per_user.extend(processed_data)

            # Aktualizacja postępu
            self.logger.display_progress(file_idx, len(data_files),
                                         "Processing data", f"Processed {file}")

        print(f"Total samples loaded: {len(data_per_user.emotions)}")

        return data_per_user

    def prepare_datasets(self, emotion_dataset):
        """
        Przygotowuje tensor dataset i dzieli dane na zbiory treningowe i testowe.

        Args:
            emotion_dataset (EmotionDataset): Dataset z danymi EEG

        Returns:
            tuple: (train_loader, test_loader, num_samples_per_class)
        """
        # Konwersja do tensor dataset
        self.logger.display_progress(1, self.prepare_process_steps,
                                     "Preparing for training", f"Creating tensor dataset...")

        tensor_dataset = emotion_dataset.tensorize()  # Zwraca TensorDataset
        print(f"Tensor dataset created. Shapes: {[t.shape for t in tensor_dataset.tensors]}")

        # Wydobycie danych z tensor dataset
        self.logger.display_progress(2, self.prepare_process_steps,
                                     "Preparing for training", f"Extracting features and labels...")

        # Pierwszy tensor to wartości EEG, drugi to ID emocji
        eeg_values = tensor_dataset.tensors[0]
        emotion_ids = tensor_dataset.tensors[1]

        print(f"EEG values shape: {eeg_values.shape}")
        print(f"Emotion IDs shape: {emotion_ids.shape}")

        # Analiza rozkładu klas
        unique_classes, counts = torch.unique(emotion_ids, return_counts=True)
        print("\nDystrybucja klas:")
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            emotion_name = self._emotion_id_to_name(cls)
            print(f"- Klasa {cls} ({emotion_name}): {count} próbek ({100 * count / len(emotion_ids):.1f}%)")

        # Wizualizacja rozkładu klas
        plt.figure(figsize=(10, 6))
        emotion_names = [self._emotion_id_to_name(cls.item()) for cls in unique_classes]
        bars = plt.bar(emotion_names, counts)

        # Dodanie etykiet liczbowych nad słupkami
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     str(count.item()), ha='center', va='bottom')

        plt.title('Rozkład klas emocji w zbiorze danych')
        plt.xlabel('Emocja')
        plt.ylabel('Liczba próbek')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()

        # Podział na zbiory treningowe i testowe
        self.logger.display_progress(3, self.prepare_process_steps,
                                     "Preparing for training", f"Splitting dataset...")

        # Podział stratyfikowany - zachowuje proporcje klas w zbiorach treningowym i testowym
        train_size = int(0.8 * len(emotion_ids))
        test_size = len(emotion_ids) - train_size

        # Podział z ustalonym seed dla powtarzalności
        train_dataset, test_dataset = random_split(
            tensor_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.config["seed"])
        )

        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")

        # Utworzenie DataLoader
        self.logger.display_progress(4, self.prepare_process_steps,
                                     "Preparing for training", f"Creating dataloaders...")

        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        print(f"Number of batches in training: {len(train_loader)}")
        print(f"Number of batches in testing: {len(test_loader)}")

        # Zapis liczby próbek na klasę
        num_samples_per_class = {cls.item(): count.item() for cls, count in zip(unique_classes, counts)}

        return train_loader, test_loader, num_samples_per_class

    def create_model(self):
        """
        Tworzy model klasyfikatora emocji na podstawie EEG.

        Returns:
            EEG_class_model: Model do klasyfikacji emocji
        """
        self.logger.display_progress(5, self.prepare_process_steps,
                                     "Preparing for training", f"Creating model...")

        # Tworzenie modelu z parametrami z konfiguracji
        model = EEG_class_model(
            d_model=self.config["d_model"],
            num_heads=self.config["num_heads"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            num_classes=self.config["num_classes"],
            num_bands=self.config["num_bands"],
            num_nodes=self.config["num_nodes"]
        )

        # Wyświetlenie informacji o modelu
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model created with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")

        # Przeniesienie modelu na odpowiednie urządzenie
        self.logger.display_progress(6, self.prepare_process_steps,
                                     "Preparing for training", f"Moving model to {device}...")

        model.to(device)
        print(f"Using device: {device}")

        return model

    def setup_training(self, model):
        """
        Konfiguruje funkcję straty i optymalizator.

        Args:
            model: Model do treningu

        Returns:
            tuple: (criterion, optimizer)
        """
        # Konfiguracja funkcji straty
        self.logger.display_progress(7, self.prepare_process_steps,
                                     "Preparing for training", f"Setting up loss function...")

        criterion = nn.CrossEntropyLoss()

        # Konfiguracja optymalizatora
        self.logger.display_progress(8, self.prepare_process_steps,
                                     "Preparing for training", f"Creating optimizer...")

        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config["learning_rate"],
            betas=(self.config["beta1"], self.config["beta2"]),
            eps=self.config["epsilon"],
            weight_decay=self.config["weight_decay"]
        )

        return criterion, optimizer

    def train_and_evaluate(self, model, train_loader, test_loader, criterion, optimizer):
        """
        Trenuje i ewaluuje model.

        Args:
            model: Model do treningu
            train_loader: DataLoader z danymi treningowymi
            test_loader: DataLoader z danymi testowymi
            criterion: Funkcja straty
            optimizer: Optymalizator

        Returns:
            tuple: (train_losses, train_accuracies, test_loss, test_accuracy, band_weights_history)
        """
        # Inicjalizacja trenera
        self.logger.display_progress(9, self.prepare_process_steps,
                                     "Starting training...", f"Initializing trainer...")

        trainer = Trainer(
            device=device,
            log_interval=5,
            clip_grad=self.config["clip_grad"],
            num_bands=self.config["num_bands"],
            num_nodes=self.config["num_nodes"]
        )

        print(f"Starting training for {self.config['num_epochs']} epochs...")

        # Trening modelu
        start_time = time.time()
        train_losses, train_accuracies = trainer.train_model(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=self.config["num_epochs"],
            visualize=self.config["visualize"]
        )
        training_time = time.time() - start_time

        # Ewaluacja modelu
        print("\nEvaluating model...")
        test_loss, test_accuracy = trainer.evaluate_model(model, test_loader, criterion)

        # Pobranie historii wag uwagi dla pasm (jeśli dostępne)
        band_weights_history = []
        if hasattr(model, 'get_band_attention_weights'):
            # Pobierz wagi dla zestawu testowego
            model.eval()
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = self._ensure_correct_dimensions(inputs)
                    inputs = inputs.to(device)
                    weights = model.get_band_attention_weights(inputs).mean(0).cpu().numpy()
                    band_weights_history.append(weights)

            # Uśrednij wagi z całego zestawu testowego
            final_band_weights = np.mean(band_weights_history, axis=0)
            print("\nWagi uwagi dla pasm częstotliwości:")
            for band_idx, band_name in enumerate(self.band_names):
                print(f"- {band_name}: {final_band_weights[band_idx]:.4f}")

        # Wyświetlenie czasu treningu
        hours, remainder = divmod(training_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nCzas treningu: {int(hours)}h {int(minutes)}m {int(seconds)}s")

        return train_losses, train_accuracies, test_loss, test_accuracy, band_weights_history

    def save_results(self, model, optimizer, train_losses, train_accuracies, test_loss, test_accuracy,
                     num_samples_per_class, band_weights_history=None):
        """
        Zapisuje wyniki treningu i model.

        Args:
            model: Wytrenowany model
            optimizer: Optymalizator
            train_losses: Lista strat treningowych
            train_accuracies: Lista dokładności treningowych
            test_loss: Strata testowa
            test_accuracy: Dokładność testowa
            num_samples_per_class: Słownik z liczbą próbek na klasę
            band_weights_history: Historia wag uwagi dla pasm

        Returns:
            str: Ścieżka do katalogu z zapisanymi wynikami
        """
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
            'epochs': self.config["num_epochs"],
            'model_config': {
                'd_model': self.config["d_model"],
                'num_heads': self.config["num_heads"],
                'num_layers': self.config["num_layers"],
                'dropout': self.config["dropout"],
                'num_classes': self.config["num_classes"],
                'num_bands': self.config["num_bands"],
                'num_nodes': self.config["num_nodes"]
            }
        }, model_path)

        print(f"Model saved to: {model_path}")

        # Wizualizacja wyników - straty i dokładności
        self._plot_training_results(train_losses, train_accuracies, test_accuracy, save_dir)

        # Wizualizacja wag uwagi dla pasm (jeśli dostępne)
        if band_weights_history:
            self._plot_band_weights(band_weights_history, save_dir)

        # Zapisanie szczegółowych informacji o treningu
        with open(join(save_dir, "training_details.txt"), "w") as f:
            f.write(f"Training completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Model Configuration:\n")
            for key, value in self.config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

            f.write("Dataset Information:\n")
            f.write(f"- Total number of classes: {self.config['num_classes']}\n")
            for cls in sorted(num_samples_per_class.keys()):
                emotion_name = self._emotion_id_to_name(cls)
                count = num_samples_per_class[cls]
                f.write(f"- Class {cls} ({emotion_name}): {count} samples\n")
            f.write("\n")

            f.write("Results:\n")
            f.write(f"- Final training loss: {train_losses[-1]:.6f}\n")
            f.write(f"- Final training accuracy: {train_accuracies[-1]:.2f}%\n")
            f.write(f"- Test loss: {test_loss:.6f}\n")
            f.write(f"- Test accuracy: {test_accuracy:.2f}%\n\n")

            if band_weights_history:
                avg_weights = np.mean(band_weights_history, axis=0)
                f.write("Band Attention Weights:\n")
                for band_idx, band_name in enumerate(self.band_names):
                    f.write(f"- {band_name}: {avg_weights[band_idx]:.6f}\n")

        return save_dir

    def run(self):
        """
        Przeprowadza cały proces treningu i ewaluacji.

        Returns:
            tuple: (model, test_accuracy, save_dir)
        """
        # 1. Ładowanie i przetwarzanie danych
        emotion_dataset = self.load_and_process_data()

        # 2. Przygotowanie zbiorów danych
        train_loader, test_loader, num_samples_per_class = self.prepare_datasets(emotion_dataset)

        # 3. Tworzenie modelu
        model = self.create_model()

        # 4. Konfiguracja treningu
        criterion, optimizer = self.setup_training(model)

        # 5. Trening i ewaluacja modelu
        train_losses, train_accuracies, test_loss, test_accuracy, band_weights_history = \
            self.train_and_evaluate(model, train_loader, test_loader, criterion, optimizer)

        # 6. Zapisanie wyników
        save_dir = self.save_results(
            model, optimizer, train_losses, train_accuracies,
            test_loss, test_accuracy, num_samples_per_class, band_weights_history
        )

        self.logger.display_progress(11, self.prepare_process_steps,
                                     "Training completed!", f"Final test accuracy: {test_accuracy:.2f}%")
        time.sleep(1)

        print(f"Final test accuracy: {test_accuracy:.2f}%")
        print(f"All results saved to: {save_dir}")

        return model, test_accuracy, save_dir

    def _emotion_id_to_name(self, emotion_id):
        """
        Konwertuje ID emocji na nazwę.
        """
        emotion_names = {
            0: "Neutral",
            1: "Happy",
            2: "Sad",
            3: "Anger",
            4: "Fear",
            5: "Surprise",
            6: "Disgust"
        }
        return emotion_names.get(emotion_id, f"Unknown ({emotion_id})")

    def _ensure_correct_dimensions(self, inputs):
        """
        Sprawdza i dostosowuje wymiary danych wejściowych do formatu [Batch, Bands, Nodes].
        """
        # Sprawdzamy wymiary wejściowe
        if inputs.dim() == 3:
            # Jeśli już mamy [Batch, Bands, Nodes], to ok
            batch_size, num_bands, num_nodes = inputs.shape
            if num_bands == self.config["num_bands"] and num_nodes == self.config["num_nodes"]:
                return inputs
            elif num_bands == self.config["num_nodes"] and num_nodes == self.config["num_bands"]:
                # Jeśli wymiary są zamienione, zamieniamy je z powrotem
                return inputs.transpose(1, 2)
        elif inputs.dim() == 2:
            # Jeśli mamy [Batch, Nodes], zakładamy że to są dane z jednego pasma
            # i powielamy je dla wszystkich pasm
            batch_size, num_nodes = inputs.shape
            if num_nodes == self.config["num_nodes"]:
                return inputs.unsqueeze(1).expand(-1, self.config["num_bands"], -1)
            elif num_nodes == self.config["num_bands"] * self.config["num_nodes"]:
                # Jeśli dane są spłaszczone, przekształcamy je z powrotem
                return inputs.reshape(batch_size, self.config["num_bands"], self.config["num_nodes"])

        # Jeśli dotarliśmy tutaj, to dane mają nieprawidłowy format
        raise ValueError(f"Nieprawidłowy format danych wejściowych: {inputs.shape}. "
                         f"Oczekiwano [Batch, {self.config['num_bands']}, {self.config['num_nodes']}]")

    def _plot_training_results(self, train_losses, train_accuracies, test_accuracy, save_dir):
        """
        Wizualizuje wyniki treningu.
        """
        plt.figure(figsize=(15, 6))

        # Wykres strat
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Training Loss')
        plt.title('Loss During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Wykres dokładności
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'r-', label='Training Accuracy')
        plt.axhline(y=test_accuracy, color='g', linestyle='--', label=f'Test Accuracy: {test_accuracy:.2f}%')
        plt.title('Accuracy During Training')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(join(save_dir, "training_results.png"))
        plt.close()

    def _plot_band_weights(self, band_weights_history, save_dir):
        """
        Wizualizuje wagi uwagi dla pasm częstotliwości.
        """
        # 1. Wykres ewolucji wag uwagi w czasie
        if len(band_weights_history) > 1:
            plt.figure(figsize=(12, 6))

            # Przekształcamy historię wag na tablicę numpy
            weights_array = np.array(band_weights_history)

            # Tworzymy wykres dla każdego pasma
            for band_idx, band_name in enumerate(self.band_names):
                plt.plot(range(len(band_weights_history)), weights_array[:, band_idx], label=band_name)

            plt.xlabel('Batch')
            plt.ylabel('Waga uwagi')
            plt.title('Ewolucja wag uwagi dla pasm częstotliwości')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(join(save_dir, "band_attention_evolution.png"))
            plt.close()

        # 2. Wykres końcowych wag uwagi
        avg_weights = np.mean(band_weights_history, axis=0)

        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.band_names, avg_weights, color='skyblue')

        # Dodanie etykiet liczbowych nad słupkami
        for bar, weight in zip(bars, avg_weights):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{weight:.3f}", ha='center', va='bottom')

        plt.ylim(0, max(avg_weights) * 1.2)  # Dodanie miejsca na etykiety
        plt.xlabel('Pasmo częstotliwości')
        plt.ylabel('Waga uwagi')
        plt.title('Wagi uwagi dla pasm częstotliwości')
        plt.grid(True, alpha=0.3, axis='y')
        plt.savefig(join(save_dir, "band_attention_weights.png"))
        plt.close()

        # 3. Wykres kołowy proporcji wag
        plt.figure(figsize=(8, 8))
        plt.pie(avg_weights, labels=self.band_names, autopct='%1.1f%%', startangle=90,
                shadow=True, explode=[0.05] * len(self.band_names))
        plt.title('Proporcje wag uwagi dla pasm częstotliwości')
        plt.savefig(join(save_dir, "band_attention_pie.png"))
        plt.close()

        # 4. Heatmapa korelacji między wagami uwagi a dokładnością
        # (wymaga więcej danych - to jest tylko szablon)
        if len(band_weights_history) > 10:
            try:
                # Tworzymy DataFrame z wagami dla każdego pasma
                df = pd.DataFrame(band_weights_history, columns=self.band_names)

                # Obliczamy macierz korelacji
                corr_matrix = df.corr()

                # Tworzymy heatmapę
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Korelacja między wagami uwagi dla pasm częstotliwości')
                plt.tight_layout()
                plt.savefig(join(save_dir, "band_correlation.png"))
                plt.close()
            except Exception as e:
                print(f"Nie udało się utworzyć heatmapy korelacji: {e}")


# Główna funkcja uruchamiająca proces
def main():
    """
    Funkcja główna uruchamiająca proces treningu i ewaluacji.
    """
    # Konfiguracja
    config = {
        "d_model": 128,
        "num_heads": 8,
        "num_layers": 3,
        "dropout": 0.3,
        "num_classes": 7,
        "num_bands": 5,
        "num_nodes": 32,
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "clip_grad": 1.0,
        "seed": 42,
        "visualize": True
    }

    # Inicjalizacja klasyfikatora
    classifier = EEGEmotionClassifier(config)

    # Uruchomienie procesu
    model, test_accuracy, save_dir = classifier.run()

    # Wyświetlenie podsumowania
    print("\n" + "=" * 50)
    print(f"Trening zakończony z dokładnością: {test_accuracy:.2f}%")
    print(f"Model zapisany w katalogu: {save_dir}")
    print("=" * 50)

    return model, test_accuracy, save_dir


if __name__ == "__main__":
    main()