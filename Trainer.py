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

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=10, visualize=True):
        """
        Trenuje model na danych treningowych z opcjonalną wizualizacją.
        """
        self.logger.display_progress(0, num_epochs, "Starting training...")
        time.sleep(1)

        model.train()
        train_losses = []
        train_accuracies = []

        # Initialize visualizer if requested
        visualizer = None
        if visualize:
            try:
                from Custom.Visualizer import SimpleEEGVisualizer
                visualizer = SimpleEEGVisualizer(model)
                print("Simple EEG visualization initialized successfully!")
            except Exception as e:
                print(f"Error initializing visualization: {e}")
                print("Training without visualization.")
                visualize = False

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5, verbose=True
        )

        for epoch in range(num_epochs):
            self.logger.display_progress(epoch, num_epochs, "Training...", f"Epoch {epoch+1}/{num_epochs}")

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

                # Zerowanie gradientów
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)

                # Sprawdzenie wymiarów wyjściowych
                if outputs.dim() == 2 and labels.dim() == 1:
                    # Standardowy przypadek: outputs [batch_size, num_classes], labels [batch_size]
                    loss = criterion(outputs, labels)
                else:
                    # Obsługa innych przypadków
                    self.logger.display_progress(batch_idx, len(train_loader), "Error",
                                                 f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")
                    raise ValueError(f"Incompatible dimensions: outputs {outputs.shape}, labels {labels.shape}")

                # Backward pass
                loss.backward()

                # Gradient clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update visualization before optimizer step
                if visualize and visualizer and batch_idx % 5 == 0:  # Update every 5 batches for performance
                    try:
                        # Calculate batch accuracy for visualization
                        _, predicted = torch.max(outputs.data, 1)
                        batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)

                        # Update and render the visualization
                        visualizer.update(loss.item(), batch_accuracy, model)
                        visualizer.render()

                        # Check if user wants to quit
                        if not visualizer.check_events():
                            print("Visualization closed. Continuing training without visualization.")
                            visualize = False
                    except Exception as e:
                        print(f"Visualization error: {e}")
                        visualize = False

                # Optymalizacja
                optimizer.step()

                # Aktualizacja statystyk
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Wyświetlanie postępu
                if batch_idx % self.log_interval == 0:
                    batch_accuracy = 100 * (predicted == labels).sum().item() / labels.size(0)
                    self.logger.display_progress(
                        batch_idx, len(train_loader),
                        "Training batch...",
                        f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1}/{len(train_loader)} "
                        f"Loss: {loss.item():.4f} Batch Accuracy: {batch_accuracy:.2f}%"
                    )

            # Statystyki dla epoki
            epoch_loss = running_loss / total
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            # Update learning rate based on loss
            scheduler.step(epoch_loss)

            print(f'Epoka [{epoch+1}/{num_epochs}], Średnia strata: {epoch_loss:.4f}, '
                  f'Dokładność: {epoch_accuracy:.2f}%')

        # Clean up
        if visualize and visualizer:
            visualizer.close()

        self.logger.display_progress(num_epochs, num_epochs, "Finishing training...")
        time.sleep(1)

        return train_losses, train_accuracies

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