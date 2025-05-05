import os
import time
import torch
from torch.nn.utils import clip_grad_norm_
from Custom.Data.Logger import Logger


class Trainer:
    """
    Klasa zarządzająca procesem treningu i ewaluacji modelu.

    Args:
        device: Urządzenie, na którym ma być wykonywany trening (CPU/GPU)
        log_interval (int): Co ile wsadów wyświetlać informacje o treningu
        clip_grad (float, optional): Maksymalna norma gradientów (None, aby wyłączyć)
    """
    def __init__(self, device, log_interval=10, clip_grad=None):
        self.device = device
        self.logger = Logger()
        self.log_interval = log_interval
        self.clip_grad = clip_grad

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=50, visualize=True):
        """
        Trenuje model na danych treningowych z opcjonalną wizualizacją.
        """
        self.logger.display_progress(0, num_epochs, "Starting training...")
        time.sleep(1)

        model.train()
        train_losses = []
        train_accuracies = []

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

        # Learning rate scheduler dla lepszej zbieżności
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.01,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=25.0,
            final_div_factor=10000.0
        )

        # Early stopping
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(num_epochs):
            self.logger.display_progress(epoch, num_epochs, "Training...", f"Epoch {epoch + 1}/{num_epochs}")

            running_loss = 0.0
            correct = 0
            total = 0

            # Iteracja po wsadach danych
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Ensure dimensions are correct
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                # Przeniesienie danych na odpowiednie urządzenie
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Augmentacja danych
                if epoch > 0:  # Zaczynamy augmentację od drugiej epoki
                    inputs = self.augment_data(inputs)

                # Zerowanie gradientów
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Sprawdzenie wymiarów wyjściowych
                if outputs.dim() == 2 and labels.dim() == 1:
                    loss = criterion(outputs, labels)
                else:
                    self.logger.display_progress(batch_idx, len(train_loader), "Error",
                                                 f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")
                    raise ValueError(f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Aktualizacja wizualizacji przed krokiem optymalizacji
                if visualize and visualizer and batch_idx % 5 == 0:  # Aktualizuj co 5 batchy dla wydajności
                    try:
                        # Oblicz dokładność dla bieżącego batcha
                        _, predicted = torch.max(outputs.data, 1)
                        batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)

                        # Aktualizuj i renderuj wizualizację
                        visualizer.update(loss.item(), batch_accuracy, model)
                        visualizer.render()

                        # Sprawdź, czy użytkownik chce zamknąć wizualizację
                        if not visualizer.check_events():
                            print("Wizualizacja zamknięta. Kontynuacja treningu bez wizualizacji.")
                            visualize = False
                    except Exception as e:
                        print(f"Błąd wizualizacji: {e}")
                        visualize = False

                # Optymalizacja
                optimizer.step()

                # Aktualizacja learning rate
                scheduler.step()

                # Aktualizacja statystyk
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Wyświetlanie postępu
                if batch_idx % self.log_interval == 0:
                    batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                    current_lr = scheduler.get_last_lr()[0]
                    self.logger.display_progress(
                        batch_idx, len(train_loader),
                        "Training batch...",
                        f"Epoch {epoch + 1}/{num_epochs} Batch {batch_idx + 1}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f} Acc: {batch_accuracy:.2f}% LR: {current_lr:.6f}"
                    )

                    # Wyświetl dystrybucję etykiet i przewidywań
                    if batch_idx == 0:
                        label_counts = torch.bincount(labels, minlength=7)
                        pred_counts = torch.bincount(predicted, minlength=7)
                        print(f"Label distribution: {label_counts}")
                        print(f"Prediction distribution: {pred_counts}")

            # Statystyki dla epoki
            epoch_loss = running_loss / total
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            print(f'Epoka [{epoch + 1}/{num_epochs}], Strata: {epoch_loss:.4f}, '
                  f'Dokładność: {epoch_accuracy:.2f}%')

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Zapisz najlepszy model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'accuracy': epoch_accuracy,
                }, 'best_model.pth')
                print(f"Zapisano najlepszy model z loss: {epoch_loss:.4f}")
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
        print(f"Załadowano najlepszy model z epoki {checkpoint['epoch'] + 1} z loss: {checkpoint['loss']:.4f}")

        return train_losses, train_accuracies

    def augment_data(self, inputs, p=0.5):
        """
        Funkcja augmentacji danych EEG.
        """
        # Kopiujemy wejście, aby nie modyfikować oryginału
        augmented = inputs.clone()

        # Dodanie szumu gaussowskiego
        if torch.rand(1).item() < p:
            noise = torch.randn_like(augmented) * 0.05
            augmented = augmented + noise

        # Maskowanie losowych elektrod
        if torch.rand(1).item() < p:
            mask = torch.rand(augmented.shape[1]) > 0.1  # Porzuć ~10% elektrod
            mask = mask.to(augmented.device)
            augmented = augmented * mask.float()

        # Skalowanie
        if torch.rand(1).item() < p:
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.2  # ±10% skalowanie
            augmented = augmented * scale

        return augmented

    def evaluate_model(self, model, test_loader, criterion):
        """
        Ewaluuje model na danych testowych.

        Args:
            model: Model do ewaluacji
            test_loader: DataLoader z danymi testowymi
            criterion: Funkcja straty

        Returns:
            tuple: (avg_loss, accuracy) - średnia strata i dokładność
        """
        self.logger.display_progress(0, len(test_loader), "Starting evaluation...")
        time.sleep(1)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                self.logger.display_progress(
                    batch_idx, len(test_loader),
                    "Evaluating...",
                    f"Batch {batch_idx+1}/{len(test_loader)}"
                )

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.display_progress(len(test_loader), len(test_loader), "Evaluating...", f"Counting accuracy...")
        time.sleep(0.5)

        accuracy = 100 * correct / total
        avg_loss = test_loss / total
        print(f'Ewaluacja - Strata: {avg_loss:.4f}, Dokładność: {accuracy:.2f}%')

        return avg_loss, accuracy