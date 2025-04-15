
import torch

class Trainer:
    def __init__(self,device):
        self.device = device

    def train_model(self, model, train_loader, criterion, optimizer, num_epochs=10):
        model.train()
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (labels, inputs) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zerowanie gradientów
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass i optymalizacja
                loss.backward()
                optimizer.step()

                # Statystyki
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Wyświetl postęp
                if (i+1) % 10 == 0:
                    print(f'Epoka [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

            epoch_loss = running_loss / len(train_loader)
            epoch_accuracy = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_accuracy)

            print(f'Epoka [{epoch+1}/{num_epochs}], Średnia strata: {epoch_loss:.4f}, '
                  f'Dokładność: {epoch_accuracy:.2f}%')

        return train_losses, train_accuracies

    # Funkcja do ewaluacji modelu
    def evaluate_model(self, model, test_loader, criterion):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for labels, inputs in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        avg_loss = test_loss / len(test_loader)
        print(f'Ewaluacja - Strata: {avg_loss:.4f}, Dokładność: {accuracy:.2f}%')

        return avg_loss, accuracy

