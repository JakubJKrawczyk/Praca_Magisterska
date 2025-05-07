import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from DataHelper import DataHelper, EmotionDataset, emotionsEnum

class Visualizer:
    def __init__(self, sample_interval_ms=100):
        """
        Initializes the Visualizer class for EEG data visualization.

        Args:
            sample_interval_ms: Time interval between consecutive samples in milliseconds
        """
        self.sample_interval_ms = sample_interval_ms
        self.emotion_data = {}
        self.color_map = {
            0: "blue",      # Neutral
            1: "green",     # Happy
            2: "purple",    # Sad
            3: "red",       # Anger
            4: "orange",    # Fear
            5: "cyan",      # Surprise
            6: "brown"      # Disgust
        }
        # Define band names and colors
        self.bands = ["delta", "theta", "alpha", "beta", "gamma"]
        self.band_colors = {
            "delta": "blue",
            "theta": "green",
            "alpha": "red",
            "beta": "purple",
            "gamma": "orange"
        }

    def load_data_for_emotion(self, dataset, emotion_ids):
        """
        Loads and filters data for specific emotions from the dataset.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_ids: Single emotion ID or list of emotion IDs to load

        Returns:
            Dictionary of EEG data for the specified emotions
        """
        # Convert single emotion ID to list for consistent processing
        if isinstance(emotion_ids, int):
            emotion_ids = [emotion_ids]

        loaded_emotions = {}

        for emotion_id in emotion_ids:
            # Find indices for the requested emotion
            indices = [i for i, e in enumerate(dataset.emotions) if e == emotion_id]

            if not indices:
                print(f"Warning: No data found for emotion ID {emotion_id} ({emotionsEnum.get(emotion_id, 'Unknown')})")
                continue

            # Extract values for the emotion
            emotion_values = [dataset.values[i] for i in indices]

            print(f"Loaded {len(emotion_values)} samples for emotion: {emotionsEnum.get(emotion_id, 'Unknown')}")
            self.emotion_data[emotion_id] = emotion_values
            loaded_emotions[emotion_id] = emotion_values

        return loaded_emotions

    def plot_time_series(self, dataset, emotion_ids=None, electrode_idx=None, title=None,
                         start_sample=None, end_sample=None, figsize=(15, 20)):
        """
        Creates a comparison of frequency bands across multiple emotions.
        Organizes plots by band first, then by emotion.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_ids: List of emotion IDs to compare
            electrode_idx: Single electrode index or list of electrode indices to visualize
            title: Optional title for the plot
            start_sample: Starting sample index (None for beginning)
            end_sample: Ending sample index (None for all)
            figsize: Figure size tuple (width, height)
        """
        # Convert single values to lists for consistent processing
        if electrode_idx is None:
            electrode_idx = [0]
        if emotion_ids is None:
            emotion_ids = [0]
        if isinstance(emotion_ids, int):
            emotion_ids = [emotion_ids]
        if isinstance(electrode_idx, int):
            electrode_idx = [electrode_idx]

        # Load data for all emotions
        self.load_data_for_emotion(dataset, emotion_ids)

        # Check if data for all emotions is loaded
        for emotion_id in emotion_ids:
            if emotion_id not in self.emotion_data:
                raise ValueError(f"Failed to load data for emotion ID {emotion_id}.")

        # Find the minimum length across all emotion datasets
        min_length = min([len(self.emotion_data[emotion_id]) for emotion_id in emotion_ids])

        # Set default start and end if not provided
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = min_length

        # Validate ranges
        if start_sample < 0:
            start_sample = 0
        if end_sample > min_length:
            end_sample = min_length
        if start_sample >= end_sample:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")

        # Calculate the number of samples
        num_samples = end_sample - start_sample

        # Create time axis (in seconds)
        time_axis = np.arange(num_samples) * (self.sample_interval_ms / 1000)

        # Close any existing figures to avoid warnings
        plt.close('all')

        # Create a grid of subplots: rows = bands, columns = emotions
        num_bands = len(self.bands)
        num_emotions = len(emotion_ids)

        # Adjust figsize based on number of subplots
        adjusted_figsize = (figsize[0], figsize[1] * num_bands / 5)

        fig, axes = plt.subplots(num_bands, num_emotions, figsize=adjusted_figsize, sharex=True)

        # Set the main title
        if not title:
            if len(electrode_idx) == 1:
                main_title = f"Comparison of Frequency Bands Across Emotions - Electrode {electrode_idx[0]+1}"
            else:
                electrode_list = ", ".join([str(e+1) for e in electrode_idx])
                main_title = f"Comparison of Frequency Bands Across Emotions - Electrodes {electrode_list}"
                main_title += f" (Samples {start_sample}-{end_sample})"
        else:
            main_title = title
        fig.suptitle(main_title, fontsize=16)

        # Generate colors for multiple electrodes if needed
        if len(electrode_idx) > 1:
            electrode_colors = plt.cm.tab10(np.linspace(0, 1, len(electrode_idx)))

        # Iterate through bands (rows) and emotions (columns)
        for i, band in enumerate(self.bands):
            for j, emotion_id in enumerate(emotion_ids):
                # Get axis (handle single row or column case)
                if num_bands == 1 and num_emotions == 1:
                    ax = axes
                elif num_bands == 1:
                    ax = axes[j]
                elif num_emotions == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]

                # Extract data for this emotion and selected range
                data = self.emotion_data[emotion_id][start_sample:end_sample]
                emotion_name = emotionsEnum.get(emotion_id, 'Unknown')

                # Set the subplot title
                ax.set_title(f"{band.capitalize()} - {emotion_name}", fontsize=11)

                # Plot each electrode's data
                for k, electrode in enumerate(electrode_idx):
                    # Extract band data for the specified electrode
                    band_data = [sample[band][electrode] for sample in data]

                    # Check for very small values and normalize if needed
                    data_max = np.max(np.abs(band_data))
                    if data_max < 1e-5 and data_max > 0:
                        print(f"Warning: Small values in {band} for {emotion_name}, electrode {electrode+1} (max={data_max})")
                        band_data = np.array(band_data) / data_max

                    # Determine color and label
                    if len(electrode_idx) > 1:
                        color = electrode_colors[k]
                        label = f"Electrode {electrode+1}"
                    else:
                        color = self.band_colors[band]
                        label = None  # No need for label with single electrode

                    # Plot the data
                    ax.plot(time_axis, band_data, color=color, linewidth=1.5, alpha=0.8, label=label)

                # Add grid
                ax.grid(True, linestyle='--', alpha=0.5)

                # Add electrode legend only in first subplot if multiple electrodes
                if len(electrode_idx) > 1 and i == 0 and j == 0:
                    ax.legend(fontsize=9, loc='upper right')

                # Add y-label only to leftmost subplots
                if j == 0:
                    ax.set_ylabel('Amplitude', fontsize=10)

                # Add x-label only to bottom row
                if i == num_bands - 1:
                    ax.set_xlabel('Time (seconds)', fontsize=10)

        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Generate a descriptive filename for saving
        emotions_str = "_".join([str(e) for e in emotion_ids])
        electrodes_str = "_".join([str(e+1) for e in electrode_idx])
        filename = f"eeg_comparison_emotions_{emotions_str}_electrodes_{electrodes_str}_samples_{start_sample}-{end_sample}.png"

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_electrode_comparison(self, dataset, emotion_ids, electrode_idx=0, band="alpha",
                                  start_sample=None, end_sample=None):
        """
        Creates a comparison plot of a specific electrode across different emotions for a specific band.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_ids: List of emotion IDs to compare
            electrode_idx: Index of the electrode to visualize
            band: Which frequency band to visualize
            start_sample: Starting sample index (None for beginning)
            end_sample: Ending sample index (None for all)
        """
        # Check if band is valid
        if band not in self.bands:
            raise ValueError(f"Invalid band: {band}. Must be one of {self.bands}")

        # Convert single emotion ID to list
        if isinstance(emotion_ids, int):
            emotion_ids = [emotion_ids]

        # Load data for all emotions
        self.load_data_for_emotion(dataset, emotion_ids)

        # Check if data for all emotions is loaded
        for emotion_id in emotion_ids:
            if emotion_id not in self.emotion_data:
                raise ValueError(f"Failed to load data for emotion ID {emotion_id}.")

        # Find minimum length across all datasets
        min_length = min([len(self.emotion_data[emotion_id]) for emotion_id in emotion_ids])

        # Set default start and end if not provided
        if start_sample is None:
            start_sample = 0
        if end_sample is None:
            end_sample = min_length

        # Validate ranges
        if start_sample < 0:
            start_sample = 0
        if end_sample > min_length:
            end_sample = min_length
        if start_sample >= end_sample:
            raise ValueError(f"Invalid sample range: start={start_sample}, end={end_sample}")

        # Create plot
        plt.figure(figsize=(12, 6))

        # Plot each emotion's data for the specified electrode and band
        for emotion_id in emotion_ids:
            # Extract band data for the specified electrode
            data = [sample[band][electrode_idx] for sample in
                    self.emotion_data[emotion_id][start_sample:end_sample]]
            time_axis = np.arange(len(data)) * (self.sample_interval_ms / 1000)

            plt.plot(time_axis, data,
                     label=f'{emotionsEnum.get(emotion_id, "Unknown")}',
                     color=self.color_map.get(emotion_id, 'gray'),
                     linewidth=2, alpha=0.8)

        plt.title(f'Electrode {electrode_idx+1} - {band.capitalize()} Band Comparison Across Emotions', fontsize=14)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"electrode_{electrode_idx+1}_{band}_comparison_samples_{start_sample}-{end_sample}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def create_animation(self, dataset, emotion_id, electrode_idx=0, duration_sec=10, fps=10):
        """
        Creates an animated visualization of EEG data over time for a specific electrode,
        showing all 5 frequency bands.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_id: ID of the emotion to visualize
            electrode_idx: Index of the electrode to visualize
            duration_sec: Duration of the animation in seconds
            fps: Frames per second
        """
        # Load data for the emotion
        self.load_data_for_emotion(dataset, emotion_id)

        # Check if data was loaded successfully
        if emotion_id not in self.emotion_data:
            raise ValueError(f"Failed to load data for emotion ID {emotion_id}.")

        # Close any existing figures to avoid the warning
        plt.close('all')

        data = self.emotion_data[emotion_id]

        # Calculate how many samples to show
        samples_per_frame = int(1000 / (self.sample_interval_ms * fps))
        total_frames = int(duration_sec * fps)
        max_samples = min(len(data), total_frames * samples_per_frame)

        # Create figure with subplots for each band
        fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"EEG Animation for {emotionsEnum.get(emotion_id, 'Unknown')} - Electrode {electrode_idx+1}",
                     fontsize=16)

        # Create lines for each band
        lines = []
        for i, band in enumerate(self.bands):
            line, = axes[i].plot([], [], color=self.band_colors[band], linewidth=2)
            lines.append(line)
            axes[i].set_title(f"{band.capitalize()} Band", fontsize=12)
            axes[i].set_ylabel('Amplitude', fontsize=10)
            axes[i].grid(True, linestyle='--', alpha=0.7)

            # Extract initial data to set y-limits
            band_data = [sample[band][electrode_idx] for sample in data[:max_samples]]
            if len(band_data) > 0:
                amplitude_max = max(np.max(np.abs(band_data)) * 1.1, 0.1)
                axes[i].set_ylim(-amplitude_max, amplitude_max)
            else:
                axes[i].set_ylim(-0.1, 0.1)

        # Set x-limits for all subplots
        for ax in axes:
            ax.set_xlim(0, self.sample_interval_ms * samples_per_frame / 1000)

        # Set x label only on the bottom subplot
        axes[-1].set_xlabel('Time (seconds)', fontsize=12)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle

        # Initialize animation data
        time_window = np.arange(samples_per_frame) * (self.sample_interval_ms / 1000)

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        def animate(i):
            start_idx = i * samples_per_frame
            if start_idx >= max_samples:
                start_idx = 0  # Loop back to beginning

            end_idx = min(start_idx + samples_per_frame, max_samples)
            actual_samples = end_idx - start_idx

            # Update lines for each band
            for j, band in enumerate(self.bands):
                band_data = [sample[band][electrode_idx] for sample in data[start_idx:end_idx]]
                lines[j].set_data(time_window[:actual_samples], band_data)

            return lines

        anim = FuncAnimation(fig, animate, frames=total_frames,
                             init_func=init, blit=True, interval=1000/fps)

        try:
            # Try with ffmpeg first
            anim.save(f'eeg_animation_emotion_{emotion_id}_electrode_{electrode_idx+1}.mp4', fps=fps)
        except TypeError:
            # Fallback to GIF with Pillow writer if ffmpeg is unavailable
            print("ffmpeg unavailable, saving as GIF instead")
            anim.save(f'eeg_animation_emotion_{emotion_id}_electrode_{electrode_idx+1}.gif', writer='pillow', fps=fps)

        plt.show()

    def create_topographic_map(self, dataset, emotion_id, sample_idx=0, band="alpha"):
        """
        Creates a topographic map visualization of electrode activity for a specific band.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_id: ID of the emotion to visualize
            sample_idx: Index of the sample to visualize
            band: Which frequency band to visualize
        """
        # Load data for the emotion
        self.load_data_for_emotion(dataset, emotion_id)

        # Check if data was loaded successfully
        if emotion_id not in self.emotion_data:
            raise ValueError(f"Failed to load data for emotion ID {emotion_id}.")

        # Check if band is valid
        if band not in self.bands:
            raise ValueError(f"Invalid band: {band}. Must be one of {self.bands}")

        # Check if sample_idx is valid
        if sample_idx >= len(self.emotion_data[emotion_id]):
            raise ValueError(f"Sample index {sample_idx} out of range. Maximum index: {len(self.emotion_data[emotion_id])-1}")

        # Get data for the sample and band
        data = [self.emotion_data[emotion_id][sample_idx][band][i] for i in range(32)]

        # Create approximate positions for 32 electrodes in a circle
        # This is just an approximation since we don't have actual positions
        angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
        radius = 0.8
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot head outline
        circle = plt.Circle((0, 0), 1, fill=False, linewidth=2)
        plt.gca().add_patch(circle)

        # Plot electrodes with color based on value
        plt.scatter(x, y, c=data, s=200, cmap='viridis',
                    vmin=np.min(data), vmax=np.max(data), zorder=2)

        # Label electrodes
        for i, (xi, yi) in enumerate(zip(x, y)):
            plt.text(xi*1.1, yi*1.1, str(i+1), fontsize=8,
                     ha='center', va='center', zorder=3)

        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('Amplitude', rotation=270, labelpad=15)

        plt.title(f'{band.capitalize()} Band Topographic Map for {emotionsEnum.get(emotion_id, "Unknown")} Emotion',
                  fontsize=14)
        plt.axis('equal')
        plt.axis('off')

        plt.savefig(f"eeg_topographic_{band}_emotion_{emotion_id}_sample_{sample_idx}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    def plot_all_bands_for_electrode(self, dataset, emotion_id, electrode_idx=0,
                                     start_sample=None, end_sample=None):
        """
        Convenience method that calls plot_time_series with appropriate parameters
        to display all 5 frequency bands for a specific electrode.

        Args:
            dataset: EmotionDataset containing the EEG data
            emotion_id: ID of the emotion to visualize
            electrode_idx: Index of the electrode to visualize
            start_sample: Starting sample index (None for beginning)
            end_sample: Ending sample index (None for all)
        """
        self.plot_time_series(
            dataset=dataset,
            emotion_ids=emotion_id,
            electrode_idx=electrode_idx,
            title=f"All Frequency Bands for Electrode {electrode_idx+1} - {emotionsEnum.get(emotion_id, 'Unknown')} Emotion",
            start_sample=start_sample,
            end_sample=end_sample
        )

# Przykład użycia:
# visualizer = Visualizer(sample_interval_ms=100)
#
# # Załadowanie datasetu
# combined_dataset = ... # Załadowanie datasetu
#
# # Wyświetlenie wszystkich pasm dla elektrody 0, emocji Happy
# visualizer.plot_all_bands_for_electrode(combined_dataset, 1, electrode_idx=0, end_sample=200)
#
# # Porównanie kilku emocji dla wybranych elektrod
# visualizer.plot_time_series(
#     dataset=combined_dataset,
#     emotion_ids=[1, 2, 3],  # Happy, Sad, Anger
#     electrode_idx=[0, 5],   # Elektrody 1 i 6
#     start_sample=100,
#     end_sample=300
# )
#
# # Porównanie pasma alpha dla elektrody 5 dla kilku emocji
# visualizer.plot_electrode_comparison(
#     dataset=combined_dataset,
#     emotion_ids=[1, 3, 5],  # Happy, Anger, Surprise
#     electrode_idx=5,
#     band="alpha",
#     start_sample=50,
#     end_sample=150
# )
#
# # Stworzenie animacji dla elektrody 0, emocji Happy
# visualizer.create_animation(combined_dataset, 1, electrode_idx=0, duration_sec=5)
#
# # Stworzenie mapy topograficznej dla pasma alpha, emocji Happy
# visualizer.create_topographic_map(combined_dataset, 1, sample_idx=10, band="alpha")