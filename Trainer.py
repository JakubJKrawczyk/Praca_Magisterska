import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from Custom.Data.Logger import Logger
import math


class Trainer:
    """
    Klasa zarządzająca procesem treningu i ewaluacji modelu EEG.
    Dostosowana do pracy z ulepszoną architekturą CNN + Transformer.

    Args:
        device: Urządzenie, na którym ma być wykonywany trening (CPU/GPU)
        log_interval (int): Co ile wsadów wyświetlać informacje o treningu
        clip_grad (float, optional): Maksymalna norma gradientów (None, aby wyłączyć)
        num_bands (int): Liczba pasm częstotliwości (domyślnie 5 - delta, theta, alpha, beta, gamma)
        num_nodes (int): Liczba elektrod (domyślnie 32)
        accumulation_steps (int): Liczba kroków do akumulacji gradientów
        warmup_pct (float): Procent epok dla warmup w schedulerze
    """

    def __init__(self, device, log_interval=10, clip_grad=1.0, num_bands=5, num_nodes=32,
                 accumulation_steps=4, warmup_pct=0.1):
        self.device = device
        self.logger = Logger()
        self.log_interval = log_interval
        self.clip_grad = clip_grad
        self.num_bands = num_bands
        self.num_nodes = num_nodes
        self.band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        self.accumulation_steps = accumulation_steps
        self.class_weights = None
        self.warmup_pct = warmup_pct

    def setup_loss_function(self, train_loader, num_classes=7):
        """
        Konfiguruje funkcję straty z wagami klas dla zbalansowania uczenia.

        Args:
            train_loader: DataLoader z danymi treningowymi
            num_classes: Liczba klas emocji

        Returns:
            torch.nn.CrossEntropyLoss: Skonfigurowana funkcja straty
        """
        print("Konfiguracja funkcji straty z wagami klas...")

        # Oblicz wagi dla klas na podstawie dystrybucji etykiet
        class_counts = torch.zeros(num_classes)
        for _, labels in train_loader:
            for label in labels:
                class_counts[label.item()] += 1

        # Zabezpieczenie przed zerami
        class_counts = torch.clamp(class_counts, min=1.0)

        # Odwrócone częstości klas jako wagi
        class_weights = 1.0 / class_counts

        # Normalizacja wag do sensownego zakresu
        class_weights = class_weights / class_weights.sum() * num_classes

        # Wyświetlenie obliczonych wag
        for cls in range(num_classes):
            emotion_name = self._emotion_id_to_name(cls)
            print(
                f"Klasa {cls} ({emotion_name}): liczba próbek = {int(class_counts[cls])}, waga = {class_weights[cls]:.4f}")

        # Przenieś na odpowiednie urządzenie
        self.class_weights = class_weights.to(self.device)

        # Utwórz funkcję straty z wagami
        return torch.nn.CrossEntropyLoss(weight=self.class_weights)

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=100, visualize=True):
        """
        Trenuje model na danych treningowych z opcjonalną wizualizacją.
        Dostosowane do pracy z ulepszoną architekturą CNN + Transformer.
        """
        self.logger.display_progress(0, num_epochs, "Starting training...")
        time.sleep(1)

        # Inicjalizacja wag modelu dla lepszej zbieżności
        # Uwaga: Model już ma własną inicjalizację wag w metodzie _init_weights()
        # Wystarczy upewnić się, że inicjalizacja jest wywołana

        # Jeśli nie przekazano funkcji straty z wagami, skonfiguruj ją automatycznie
        if self.class_weights is None and isinstance(criterion, torch.nn.CrossEntropyLoss):
            criterion = self.setup_loss_function(train_loader, num_classes=7)

        model.train()
        train_losses = []
        train_accuracies = []
        band_weights_history = []  # Historia wag uwagi dla pasm

        # Inicjalizacja wizualizatora
        visualizer = None
        if visualize:
            try:
                from Custom.Visualizer import SimpleEEGVisualizer
                visualizer = SimpleEEGVisualizer(model)
                print("Wizualizacja zainicjalizowana!")
            except Exception as e:
                print(f"Błąd inicjalizacji wizualizacji: {e}")
                visualize = False

        # Scheduler uczenia z warm-up i cosine annealing - ZAMIENIONY NA BARDZIEJ ZAAWANSOWANY
        def lr_lambda(epoch):
            # Warm-up przez pierwsze warmup_pct epok
            warmup_epochs = int(self.warmup_pct * num_epochs)
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            # Kosinus annealing po warm-up
            return 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # Early stopping
        best_loss = float('inf')
        best_accuracy = 0.0
        patience = 20
        patience_counter = 0

        for epoch in range(num_epochs):
            self.logger.display_progress(epoch, num_epochs, "Training...", f"Epoch {epoch + 1}/{num_epochs}")

            running_loss = 0.0
            correct = 0
            total = 0
            epoch_band_weights = []

            # Zerowanie gradientów na początku epoki
            optimizer.zero_grad()

            # Iteracja po wsadach danych
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Sprawdzenie i dostosowanie wymiarów danych wejściowych
                inputs = self._ensure_correct_dimensions(inputs)

                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                # Przeniesienie danych na odpowiednie urządzenie
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Uwaga: Normalizację per-sample wykonuje teraz sam model w metodzie forward
                # poprzez wywołanie _normalize_per_sample(x)

                # Augmentacja danych - ZMODYFIKOWANE PARAMETRY
                if epoch > 0:  # Zaczynamy augmentację od drugiej epoki
                    inputs = self.augment_data(inputs, p=0.3)  # Zredukowane prawdopodobieństwo

                # Forward pass
                outputs = model(inputs)

                # Sprawdzenie wymiarów wyjściowych
                if outputs.dim() == 2 and labels.dim() == 1:
                    loss = criterion(outputs, labels)
                else:
                    self.logger.display_progress(batch_idx, len(train_loader), "Error",
                                                 f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")
                    raise ValueError(f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")

                # Skalowanie straty dla akumulacji gradientów
                loss = loss / self.accumulation_steps

                # Backward pass
                loss.backward()

                # Aktualizacja statystyk
                running_loss += loss.item() * inputs.size(0) * self.accumulation_steps  # Korekta dla akumulacji
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Aktualizacja wag tylko co accumulation_steps batchy
                if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
                    # Gradient clipping
                    if self.clip_grad is not None:
                        clip_grad_norm_(model.parameters(), self.clip_grad)

                    # Optymalizacja
                    optimizer.step()
                    optimizer.zero_grad()

                    # Aktualizacja wizualizacji po kroku optymalizacji
                    if visualize and visualizer and (batch_idx // self.accumulation_steps) % 5 == 0:
                        try:
                            # Oblicz dokładność dla bieżącego batcha
                            batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)

                            # Pobierz wagi uwagi dla pasm (jeśli model ma taką metodę)
                            band_attention = None
                            if hasattr(model, 'get_band_attention_weights'):
                                band_attention = model.get_band_attention_weights(inputs).detach().cpu().numpy()

                            # Aktualizuj i renderuj wizualizację
                            visualizer.update(loss.item() * self.accumulation_steps, batch_accuracy, model,
                                              band_attention=band_attention)
                            visualizer.render()

                            # Sprawdź, czy użytkownik chce zamknąć wizualizację
                            if not visualizer.check_events():
                                print("Wizualizacja zamknięta. Kontynuacja treningu bez wizualizacji.")
                                visualize = False
                        except Exception as e:
                            print(f"Błąd wizualizacji: {e}")
                            visualize = False

                # Pobierz i zapisz wagi uwagi dla pasm (jeśli dostępne)
                if hasattr(model, 'get_band_attention_weights') and batch_idx % (5 * self.accumulation_steps) == 0:
                    with torch.no_grad():
                        # Uwaga: format wag uwagi może być inny w nowym modelu (seq_len = bands+nodes)
                        band_weights = model.get_band_attention_weights(inputs)
                        # Sprawdź wielkość pierwszego wymiaru (długość sekwencji)
                        seq_len = band_weights.shape[1]
                        if seq_len == self.num_bands + self.num_nodes:
                            # Jeśli sekwencja to [bands+nodes], pobierz tylko część odpowiadającą pasmom
                            band_weights = band_weights[:, :self.num_bands]

                        batch_band_weights = band_weights.mean(0).detach().cpu().numpy()
                        epoch_band_weights.append(batch_band_weights)

                # Wyświetlanie postępu
                if batch_idx % self.log_interval == 0:
                    batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                    current_lr = scheduler.get_last_lr()[0]
                    self.logger.display_progress(
                        batch_idx, len(train_loader),
                        "Training batch...",
                        f"Epoch {epoch + 1}/{num_epochs} Batch {batch_idx + 1}/{len(train_loader)} "
                        f"Loss: {loss.item() * self.accumulation_steps:.4f} Acc: {batch_accuracy:.2f}% LR: {current_lr:.6f}"
                    )

                    # Wyświetl dystrybucję etykiet i przewidywań
                    if batch_idx == 0:
                        label_counts = torch.bincount(labels, minlength=7)
                        pred_counts = torch.bincount(predicted, minlength=7)
                        print(f"Label distribution: {label_counts}")
                        print(f"Prediction distribution: {pred_counts}")

                        # Wyświetl wagi uwagi dla pasm, jeśli dostępne
                        if hasattr(model, 'get_band_attention_weights') and len(epoch_band_weights) > 0:
                            weight_str = " | ".join(
                                [f"{name}: {weight:.3f}" for name, weight in
                                 zip(self.band_names, epoch_band_weights[-1])])
                            print(f"Band weights: {weight_str}")

            # Uśrednij wagi uwagi dla epoki
            if epoch_band_weights:
                avg_band_weights = np.mean(epoch_band_weights, axis=0)
                band_weights_history.append(avg_band_weights)

                # Wyświetl wagi uwagi dla epoki
                weight_str = " | ".join(
                    [f"{name}: {weight:.3f}" for name, weight in zip(self.band_names, avg_band_weights)])
                print(f"Epoch avg band weights: {weight_str}")

            # Statystyki dla epoki
            epoch_loss = running_loss / total
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            print(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {epoch_loss:.4f}, '
                  f'Dokładność: {epoch_accuracy:.2f}%')

            # Wyświetl wykres wag uwagi dla pasm co 5 epok
            if epoch % 5 == 0 and band_weights_history:
                self._plot_band_weights(band_weights_history, epoch)

            # Aktualizuj learning rate na koniec epoki
            scheduler.step()

            # Early stopping - monitoruje zarówno loss jak i accuracy
            if epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_loss = epoch_loss  # Nadal zapisuj stratę dla referencji
                patience_counter = 0
                # Zapisz najlepszy model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_accuracy,
                }, 'best_model.pth')
                print(f"Zapisano najlepszy model z dokładnością: {epoch_accuracy:.2f}% i stratą: {epoch_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping po {epoch + 1} epokach!")
                    break

        # Zamknij wizualizację po zakończeniu
        if visualize and visualizer:
            visualizer.close()

        self.logger.display_progress(num_epochs, num_epochs, "Finishing training...")
        time.sleep(1)

        # Załaduj najlepszy model
        checkpoint = torch.load('best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(
            f"Załadowano najlepszy model z epoki {checkpoint['epoch'] + 1} z dokładnością: {checkpoint['accuracy']:.2f}%")

        # Jeśli mamy historię wag uwagi, zapisz wykres
        if band_weights_history:
            self._plot_band_weights(band_weights_history, num_epochs, save=True)

        return train_losses, train_accuracies

    def _ensure_correct_dimensions(self, inputs):
        """
        Sprawdza i dostosowuje wymiary danych wejściowych do formatu [Batch, Bands, Nodes].
        """
        # Sprawdzamy wymiary wejściowe
        if inputs.dim() == 3:
            # Jeśli już mamy [Batch, Bands, Nodes], to ok
            batch_size, num_bands, num_nodes = inputs.shape
            if num_bands == self.num_bands and num_nodes == self.num_nodes:
                return inputs
            elif num_bands == self.num_nodes and num_nodes == self.num_bands:
                # Jeśli wymiary są zamienione, zamieniamy je z powrotem
                return inputs.transpose(1, 2)
        elif inputs.dim() == 2:
            # Jeśli mamy [Batch, Nodes], zakładamy że to są dane z jednego pasma
            # i powielamy je dla wszystkich pasm
            batch_size, num_nodes = inputs.shape
            if num_nodes == self.num_nodes:
                return inputs.unsqueeze(1).expand(-1, self.num_bands, -1)
            elif num_nodes == self.num_bands * self.num_nodes:
                # Jeśli dane są spłaszczone, przekształcamy je z powrotem
                return inputs.reshape(batch_size, self.num_bands, self.num_nodes)

        # Jeśli dotarliśmy tutaj, to dane mają nieprawidłowy format
        raise ValueError(f"Nieprawidłowy format danych wejściowych: {inputs.shape}. "
                         f"Oczekiwano [Batch, {self.num_bands}, {self.num_nodes}]")

    def augment_data(self, inputs, p=0.3):
        """
        Funkcja augmentacji danych EEG dla formatu [Batch, Bands, Nodes].
        Wykonuje augmentację zachowującą strukturę pasm częstotliwości.
        """
        # Kopiujemy wejście, aby nie modyfikować oryginału
        augmented = inputs.clone()
        batch_size, num_bands, num_nodes = augmented.shape

        # Dodanie szumu gaussowskiego (niezależnie dla każdego pasma)
        if torch.rand(1).item() < p:
            noise_scale = 0.02  # ZREDUKOWANY SZUM
            noise = torch.randn_like(augmented) * noise_scale
            augmented = augmented + noise

        # Maskowanie losowych elektrod (wspólne dla wszystkich pasm)
        if torch.rand(1).item() < p:
            # Tworzymy maskę dla elektrod (wspólną dla wszystkich pasm)
            mask = torch.rand(num_nodes) > 0.05  # ZREDUKOWANE MASKOWANIE (5% zamiast 10%)
            mask = mask.to(augmented.device)

            # Aplikujemy maskę do wszystkich pasm
            for band in range(num_bands):
                augmented[:, band] = augmented[:, band] * mask.float()

        # Skalowanie (niezależne dla każdego pasma)
        if torch.rand(1).item() < p:
            for band in range(num_bands):
                # Losowe skalowanie ±5% dla każdego pasma (ZREDUKOWANE)
                scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
                augmented[:, band] = augmented[:, band] * scale

        # Przesunięcie fazowe (mieszanie sąsiednich elektrod w obrębie pasma)
        if torch.rand(1).item() < p:
            for band in range(num_bands):
                # Tworzymy jądro konwolucji 3x1 dla przesunięcia fazowego
                kernel = torch.tensor([0.2, 0.6, 0.2]).to(augmented.device)
                kernel = kernel.view(1, 1, 3)  # [out_channels, in_channels, kernel_size]

                # Przekształcamy dane do formatu [batch, channel=1, nodes]
                band_data = augmented[:, band].unsqueeze(1)

                # Stosujemy konwolucję 1D z paddingiem (odbijanie na granicy)
                padded = torch.nn.functional.pad(band_data, (1, 1), mode='reflect')
                convolved = torch.nn.functional.conv1d(padded, kernel)

                # Zapisujemy wynik
                augmented[:, band] = convolved.squeeze(1)

        return augmented

    def _plot_band_weights(self, band_weights_history, current_epoch, save=False):
        """
        Tworzy wykres wag uwagi dla pasm częstotliwości w czasie treningu.
        """
        try:
            plt.figure(figsize=(10, 6))
            epochs = range(len(band_weights_history))

            # Przekształcamy historię wag na tablicę numpy
            weights_array = np.array(band_weights_history)

            # Tworzymy wykres dla każdego pasma
            for band_idx, band_name in enumerate(self.band_names):
                if band_idx < weights_array.shape[1]:  # Zabezpieczenie przed indeksem poza zakresem
                    plt.plot(epochs, weights_array[:, band_idx], label=band_name)

            plt.xlabel('Epoka')
            plt.ylabel('Waga uwagi')
            plt.title(f'Ewolucja wag uwagi dla pasm częstotliwości (epoka {current_epoch})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save:
                plt.savefig('band_attention_weights.png')
                print("Zapisano wykres wag uwagi dla pasm do pliku band_attention_weights.png")
            else:
                plt.show(block=False)
                plt.pause(2)  # Wyświetlaj przez 2 sekundy
                plt.close()

        except Exception as e:
            print(f"Błąd podczas tworzenia wykresu wag uwagi: {e}")

    def evaluate_model(self, model, test_loader, criterion):
        """
        Ewaluuje model na danych testowych.
        Dostosowane do pracy z ulepszoną architekturą CNN + Transformer.
        """
        self.logger.display_progress(0, len(test_loader), "Starting evaluation...")
        time.sleep(1)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        # Przygotowanie statystyk dla poszczególnych klas emocji
        emotion_correct = torch.zeros(7)  # 7 klas emocji
        emotion_total = torch.zeros(7)

        # Historia wag uwagi dla pasm podczas ewaluacji
        test_band_weights = []

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                self.logger.display_progress(
                    batch_idx, len(test_loader),
                    "Evaluating...",
                    f"Batch {batch_idx + 1}/{len(test_loader)}"
                )

                # Dostosowanie wymiarów wejściowych
                inputs = self._ensure_correct_dimensions(inputs)

                # Uwaga: Normalizację per-sample wykonuje teraz sam model
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Aktualizacja statystyk dla poszczególnych klas emocji
                for cls in range(7):
                    cls_mask = (labels == cls)
                    if cls_mask.sum() > 0:
                        emotion_total[cls] += cls_mask.sum().item()
                        emotion_correct[cls] += ((predicted == cls) & cls_mask).sum().item()

                # Pobierz wagi uwagi dla pasm (jeśli dostępne)
                if hasattr(model, 'get_band_attention_weights'):
                    band_weights = model.get_band_attention_weights(inputs)
                    # Sprawdź wielkość pierwszego wymiaru (długość sekwencji)
                    seq_len = band_weights.shape[1]
                    if seq_len == self.num_bands + self.num_nodes:
                        # Jeśli sekwencja to [bands+nodes], pobierz tylko część odpowiadającą pasmom
                        band_weights = band_weights[:, :self.num_bands]

                    batch_band_weights = band_weights.mean(0).detach().cpu().numpy()
                    test_band_weights.append(batch_band_weights)

        self.logger.display_progress(len(test_loader), len(test_loader), "Evaluating...", f"Counting accuracy...")
        time.sleep(0.5)

        accuracy = 100 * correct / total
        avg_loss = test_loss / total
        print(f'Ewaluacja - Strata: {avg_loss:.4f}, Dokładność: {accuracy:.2f}%')

        # Analiza macierzy pomyłek
        if total > 0:
            print("\nAnaliza klasyfikacji:")
            confusion_matrix = torch.zeros(7, 7, dtype=torch.int)

            # Pobierz wszystkie predykcje i etykiety dla całego zbioru testowego
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = self._ensure_correct_dimensions(inputs)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)

                    # Utwórz macierz pomyłek
                    for i in range(len(labels)):
                        confusion_matrix[labels[i], predicted[i]] += 1

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Wyświetl macierz pomyłek
            print("\nMacierz pomyłek:")
            print("   ", end="")
            for i in range(7):
                print(f"{i:4}", end="")
            print("\n   +", "----" * 7)

            for i in range(7):
                print(f"{i} |", end="")
                for j in range(7):
                    print(f"{confusion_matrix[i][j]:4}", end="")
                print(f" | {self._emotion_id_to_name(i)}")

            # Oblicz precision, recall i F1 dla każdej klasy
            print("\nMetryki dla poszczególnych klas:")
            print("Klasa | Precision | Recall  | F1 Score")
            print("-" * 40)

            for cls in range(7):
                if emotion_total[cls] > 0:
                    # True Positives: Prawidłowe przewidywania dla klasy cls
                    tp = confusion_matrix[cls, cls].item()

                    # False Positives: Suma kolumny cls - TP
                    fp = confusion_matrix[:, cls].sum().item() - tp

                    # False Negatives: Suma wiersza cls - TP
                    fn = confusion_matrix[cls, :].sum().item() - tp

                    # Precision: TP / (TP + FP)
                    precision = tp / (tp + fp + 1e-8)

                    # Recall: TP / (TP + FN)
                    recall = tp / (tp + fn + 1e-8)

                    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

                    print(f"{cls:5} | {precision:.4f}   | {recall:.4f} | {f1:.4f}   | {self._emotion_id_to_name(cls)}")

        # Wyświetl dokładność dla poszczególnych klas emocji
        print("\nDokładność dla poszczególnych emocji:")
        for cls in range(7):
            if emotion_total[cls] > 0:
                cls_accuracy = 100 * emotion_correct[cls] / emotion_total[cls]
                print(
                    f"- {cls} ({self._emotion_id_to_name(cls)}): {cls_accuracy:.2f}% ({int(emotion_correct[cls])}/{int(emotion_total[cls])})")

        # Wyświetl średnie wagi uwagi dla pasm
        if test_band_weights:
            avg_test_band_weights = np.mean(test_band_weights, axis=0)
            print("\nŚrednie wagi uwagi dla pasm podczas ewaluacji:")
            for band_idx, band_name in enumerate(self.band_names):
                if band_idx < len(avg_test_band_weights):  # Zabezpieczenie przed indeksem poza zakresem
                    print(f"- {band_name}: {avg_test_band_weights[band_idx]:.4f}")

            # Sortuj pasma według wagi uwagi
            sorted_indices = np.argsort(avg_test_band_weights)[::-1]  # Malejąco
            print("\nPasma według istotności:")
            for i, idx in enumerate(sorted_indices):
                if idx < len(self.band_names):  # Zabezpieczenie przed indeksem poza zakresem
                    print(f"{i + 1}. {self.band_names[idx]}: {avg_test_band_weights[idx]:.4f}")

        return avg_loss, accuracy

    def _init_model_weights(self, model):
        """
        Inicjalizuje wagi modelu dla lepszej zbieżności.
        Uwaga: Nowa wersja modelu ma własną metodę _init_weights(),
        więc ta funkcja jest pozostawiona dla kompatybilności.
        """
        if hasattr(model, '_init_weights'):
            # Jeśli model ma własną metodę inicjalizacji, wywołaj ją
            print("Używanie wbudowanej inicjalizacji wag modelu...")
            return

        print("Inicjalizacja wag modelu...")
        for name, p in model.named_parameters():
            if 'weight' in name:
                if len(p.shape) >= 2:
                    # Inicjalizacja Kaiming dla warstw liniowych i konwolucyjnych
                    torch.nn.init.kaiming_normal_(p, mode='fan_out', nonlinearity='relu')
                else:
                    # Inicjalizacja dla biasów i innych parametrów 1D
                    torch.nn.init.normal_(p, mean=0.0, std=0.01)
            elif 'bias' in name:
                torch.nn.init.zeros_(p)

    def _normalize_input_batch(self, inputs):
        """
        Normalizuje dane wejściowe dla każdej próbki.
        Uwaga: Nowa wersja modelu ma własną metodę _normalize_per_sample(),
        więc ta funkcja jest pozostawiona dla kompatybilności.
        """
        # Kształt danych wejściowych
        batch_size, num_bands, num_nodes = inputs.shape

        # Reshape do [samples, bands*nodes]
        flat_inputs = inputs.reshape(batch_size, -1)

        # Normalizacja Z-score dla każdej próbki
        mean = flat_inputs.mean(dim=1, keepdim=True)
        std = flat_inputs.std(dim=1, keepdim=True) + 1e-8
        normalized = (flat_inputs - mean) / std

        # Reshape z powrotem
        return normalized.reshape(inputs.shape)

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