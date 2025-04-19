import pygame
import numpy as np
import time
import os

class SimpleEEGVisualizer:
    """
    A simple and robust visualization system for EEG model training
    that uses only pygame's native drawing capabilities.
    """
    def __init__(self, model, width=1200, height=800):
        # Initialize pygame
        pygame.init()
        pygame.display.set_caption("EEG Training Visualization")
        self.screen = pygame.display.set_mode((width, height))
        self.width = width
        self.height = height
        self.model = model

        # Font setup
        self.font = pygame.font.SysFont('Arial', 18)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)

        # Colors
        self.bg_color = (20, 20, 30)
        self.grid_color = (50, 50, 70)
        self.text_color = (220, 220, 220)
        self.title_color = (180, 180, 255)
        self.loss_color = (255, 100, 100)
        self.acc_color = (100, 200, 255)
        self.grad_colors = [
            (255, 160, 100),  # Orange
            (100, 255, 160),  # Green
            (160, 100, 255),  # Purple
            (255, 255, 100),  # Yellow
            (100, 200, 255),  # Light blue
            (255, 100, 200)   # Pink
        ]

        # Data storage
        self.losses = []
        self.accuracies = []
        self.batch_counter = 0
        self.gradient_history = {}

        print("Simple EEG Visualizer initialized successfully!")

    def render_text(self, text, position, color=None, font=None):
        """Draw text on the screen"""
        if color is None:
            color = self.text_color
        if font is None:
            font = self.font

        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, position)

    def update(self, loss, accuracy, model):
        """Update visualization data with new training information"""
        self.batch_counter += 1

        # Store metrics
        self.losses.append(loss)
        self.accuracies.append(accuracy / 100.0)  # Convert to 0-1 range

        # Keep the history limited to avoid memory issues
        if len(self.losses) > 1000:
            self.losses = self.losses[-1000:]
            self.accuracies = self.accuracies[-1000:]

        # Store gradient information
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Calculate gradient norm
                grad_norm = param.grad.norm().item()

                # Store only for important layers (weights, not biases)
                if 'weight' in name:
                    # Create a simplified name
                    parts = name.split('.')
                    if len(parts) > 2:
                        short_name = f"{parts[-3][:5]}.{parts[-2][:5]}.{parts[-1][:5]}"
                    else:
                        short_name = name

                    if short_name not in self.gradient_history:
                        self.gradient_history[short_name] = []

                    history = self.gradient_history[short_name]
                    history.append(grad_norm)

                    # Limit history length
                    if len(history) > 200:
                        self.gradient_history[short_name] = history[-200:]

    def draw_metrics_plot(self, x, y, width, height):
        """Draw loss and accuracy plot"""
        # Draw plot background and border
        plot_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, plot_rect)
        pygame.draw.rect(self.screen, self.grid_color, plot_rect, 1)

        # Draw title
        self.render_text("Training Progress",
                         (x + width // 2 - 70, y - 30),
                         self.title_color,
                         self.title_font)

        # Draw grid
        for i in range(1, 10):
            # Horizontal grid lines
            y_pos = y + i * height // 10
            pygame.draw.line(self.screen, self.grid_color,
                             (x, y_pos),
                             (x + width, y_pos), 1)

            # Vertical grid lines
            x_pos = x + i * width // 10
            pygame.draw.line(self.screen, self.grid_color,
                             (x_pos, y),
                             (x_pos, y + height), 1)

        # Draw loss values if available
        if len(self.losses) > 1:
            # Determine y-axis scaling
            max_loss = max(self.losses)
            min_loss = min(self.losses)
            loss_range = max(0.1, max_loss - min_loss)

            # Draw y-axis labels for loss
            for i in range(0, 11, 2):
                label_value = min_loss + (i / 10) * loss_range
                label_y = y + height - (i * height // 10)
                self.render_text(f"{label_value:.2f}", (x - 50, label_y - 10))

            # Select points to display (for efficiency)
            display_points = min(100, len(self.losses))
            stride = max(1, len(self.losses) // display_points)
            points = self.losses[::stride][-display_points:]

            # Create points for the loss line
            loss_points = []
            for i, loss_val in enumerate(points):
                point_x = x + i * width // len(points)
                # Normalize to fit in plot area
                normalized_val = (loss_val - min_loss) / loss_range
                point_y = y + height - int(normalized_val * height * 0.8)
                loss_points.append((point_x, point_y))

            # Draw the loss line
            if len(loss_points) > 1:
                pygame.draw.lines(self.screen, self.loss_color, False, loss_points, 2)

        # Draw accuracy values if available
        if len(self.accuracies) > 1:
            # Select points to display
            display_points = min(100, len(self.accuracies))
            stride = max(1, len(self.accuracies) // display_points)
            points = self.accuracies[::stride][-display_points:]

            # Create points for the accuracy line
            acc_points = []
            for i, acc_val in enumerate(points):
                point_x = x + i * width // len(points)
                # Accuracy is already normalized (0-1)
                point_y = y + height - int(acc_val * height * 0.8)
                acc_points.append((point_x, point_y))

            # Draw the accuracy line
            if len(acc_points) > 1:
                pygame.draw.lines(self.screen, self.acc_color, False, acc_points, 2)

            # Draw y-axis labels for accuracy (right side)
            for i in range(0, 11, 2):
                label_value = i / 10
                label_y = y + height - (i * height // 10)
                self.render_text(f"{label_value:.1f}", (x + width + 10, label_y - 10), self.acc_color)

        # Draw legend
        legend_x = x + width - 150
        legend_y = y + 20

        # Loss legend
        pygame.draw.line(self.screen, self.loss_color,
                         (legend_x, legend_y),
                         (legend_x + 20, legend_y), 2)
        self.render_text("Loss", (legend_x + 30, legend_y - 10), self.loss_color)

        # Accuracy legend
        pygame.draw.line(self.screen, self.acc_color,
                         (legend_x, legend_y + 25),
                         (legend_x + 20, legend_y + 25), 2)
        self.render_text("Accuracy", (legend_x + 30, legend_y + 15), self.acc_color)

    def draw_gradient_plot(self, x, y, width, height):
        """Draw gradient magnitudes plot"""
        # Draw plot background and border
        plot_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, plot_rect)
        pygame.draw.rect(self.screen, self.grid_color, plot_rect, 1)

        # Draw title
        self.render_text("Gradient Magnitudes (Log Scale)",
                         (x + width // 2 - 120, y - 30),
                         self.title_color,
                         self.title_font)

        # Draw grid
        for i in range(1, 10):
            # Horizontal grid lines
            y_pos = y + i * height // 10
            pygame.draw.line(self.screen, self.grid_color,
                             (x, y_pos),
                             (x + width, y_pos), 1)

            # Vertical grid lines
            x_pos = x + i * width // 10
            pygame.draw.line(self.screen, self.grid_color,
                             (x_pos, y),
                             (x_pos, y + height), 1)

        # Select a few important gradient histories to display
        if self.gradient_history:
            # Get the 5 most important layers (with highest recent gradient)
            important_layers = []
            for name, history in self.gradient_history.items():
                if history:
                    important_layers.append((name, history[-1]))

            important_layers.sort(key=lambda x: x[1], reverse=True)
            important_layers = important_layers[:6]  # Top 6 layers

            # Draw y-axis labels (log scale)
            min_val = 1e-5
            max_val = 10.0
            for i in range(6):
                label_value = min_val * (max_val/min_val)**(i/5)
                label_y = y + height - int((i/5) * height)
                self.render_text(f"{label_value:.5f}", (x - 75, label_y - 10))

            # Draw each layer's gradient history
            for i, (name, _) in enumerate(important_layers):
                history = self.gradient_history[name]

                # Select points to display
                display_points = min(100, len(history))
                stride = max(1, len(history) // display_points)
                points = history[::stride][-display_points:]

                # Create points for the gradient line
                grad_points = []
                for j, grad_val in enumerate(points):
                    point_x = x + j * width // len(points)

                    # Log scale mapping
                    if grad_val <= 0:
                        grad_val = min_val  # Handle zero or negative values

                    # Normalize to [0, 1] on log scale
                    log_min = np.log10(min_val)
                    log_max = np.log10(max_val)
                    normalized_val = (np.log10(grad_val) - log_min) / (log_max - log_min)
                    normalized_val = max(0, min(1, normalized_val))  # Clamp to [0, 1]

                    point_y = y + height - int(normalized_val * height)
                    grad_points.append((point_x, point_y))

                # Draw the gradient line
                color = self.grad_colors[i % len(self.grad_colors)]
                if len(grad_points) > 1:
                    pygame.draw.lines(self.screen, color, False, grad_points, 2)

                # Draw legend
                legend_x = x + 10
                legend_y = y + 20 + i * 25
                pygame.draw.line(self.screen, color,
                                 (legend_x, legend_y),
                                 (legend_x + 20, legend_y), 2)
                self.render_text(f"{name}", (legend_x + 30, legend_y - 10), color)

    def draw_gradient_info(self, x, y, width, height):
        """Draw text information about gradients"""
        # Draw section background and border
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, rect)
        pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        # Draw title
        self.render_text("Latest Gradient Values",
                         (x + width // 2 - 90, y - 30),
                         self.title_color,
                         self.title_font)

        # Draw gradient information
        if self.gradient_history:
            info_x = x + 20
            info_y = y + 20

            self.render_text("Layer", (info_x, info_y))
            self.render_text("Gradient Norm", (info_x + 200, info_y))
            self.render_text("Status", (info_x + 350, info_y))

            info_y += 30

            # Sort by gradient norm
            grad_items = []
            for name, history in self.gradient_history.items():
                if history:
                    grad_items.append((name, history[-1]))

            grad_items.sort(key=lambda x: x[1], reverse=True)

            # Display the information
            for i, (name, grad_norm) in enumerate(grad_items):
                # Skip if we've reached the bottom of the area
                if info_y > y + height - 20:
                    break

                # Determine color based on gradient magnitude
                if grad_norm > 1.0:
                    color = (255, 150, 150)  # Red for large gradients
                    status = "Too high!"
                elif grad_norm < 0.0001:
                    color = (150, 150, 255)  # Blue for small gradients
                    status = "Too low"
                else:
                    color = (150, 255, 150)  # Green for good gradients
                    status = "Good"

                self.render_text(name, (info_x, info_y))
                self.render_text(f"{grad_norm:.6f}", (info_x + 200, info_y), color)
                self.render_text(status, (info_x + 350, info_y), color)

                info_y += 25

    def draw_stats(self, x, y, width, height):
        """Draw current training statistics"""
        # Draw section background and border
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, self.bg_color, rect)
        pygame.draw.rect(self.screen, self.grid_color, rect, 1)

        # Draw title
        self.render_text("Training Statistics",
                         (x + width // 2 - 80, y - 30),
                         self.title_color,
                         self.title_font)

        # Draw stats
        info_x = x + 20
        info_y = y + 20

        self.render_text(f"Batch Count: {self.batch_counter}", (info_x, info_y))
        info_y += 30

        if self.losses:
            current_loss = self.losses[-1]
            current_acc = self.accuracies[-1] * 100

            # Calculate trends
            loss_trend = "—"
            acc_trend = "—"

            if len(self.losses) > 10:
                recent_losses = self.losses[-10:]
                if recent_losses[0] > recent_losses[-1]:
                    loss_trend = "▼ Decreasing"
                elif recent_losses[0] < recent_losses[-1]:
                    loss_trend = "▲ Increasing"
                else:
                    loss_trend = "— Stable"

                recent_accs = self.accuracies[-10:]
                if recent_accs[0] < recent_accs[-1]:
                    acc_trend = "▲ Improving"
                elif recent_accs[0] > recent_accs[-1]:
                    acc_trend = "▼ Declining"
                else:
                    acc_trend = "— Stable"

            # Display current loss
            self.render_text(f"Current Loss: {current_loss:.6f}", (info_x, info_y))
            self.render_text(f"Trend: {loss_trend}", (info_x + 250, info_y))
            info_y += 30

            # Display current accuracy
            self.render_text(f"Current Accuracy: {current_acc:.2f}%", (info_x, info_y))
            self.render_text(f"Trend: {acc_trend}", (info_x + 250, info_y))
            info_y += 30

            # Display min/max values
            if len(self.losses) > 1:
                min_loss = min(self.losses)
                max_loss = max(self.losses)
                min_acc = min(self.accuracies) * 100
                max_acc = max(self.accuracies) * 100

                self.render_text(f"Loss Range: {min_loss:.4f} - {max_loss:.4f}", (info_x, info_y))
                info_y += 30
                self.render_text(f"Accuracy Range: {min_acc:.2f}% - {max_acc:.2f}%", (info_x, info_y))

    def render(self):
        """Render the complete visualization"""
        # Fill the background
        self.screen.fill(self.bg_color)

        # Draw main title
        title = "EEG Emotion Classification - Training Visualization"
        self.render_text(title, (self.width // 2 - 200, 10),
                         self.title_color, self.title_font)

        # Layout the visualization components
        self.draw_metrics_plot(50, 80, 500, 300)
        self.draw_gradient_plot(600, 80, 550, 300)
        self.draw_gradient_info(50, 450, 550, 300)
        self.draw_stats(650, 450, 500, 300)

        # Update the display
        pygame.display.flip()

    def check_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()
                    return False
                elif event.key == pygame.K_s:
                    # Save screenshot
                    self.save_screenshot()
        return True

    def save_screenshot(self, directory="visualization"):
        """Save current visualization as an image"""
        if not os.path.exists(directory):
            os.makedirs(directory)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{directory}/eeg_vis_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved to {filename}")

    def close(self):
        """Clean up resources"""
        pygame.quit()