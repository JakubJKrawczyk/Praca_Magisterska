import os

import numpy as np

from DataHelper import DataHelper
from DataHelper import EmotionDataset
from Visualizer import Visualizer

def get_data_for_emotion(ds : EmotionDataset, target_emotion_id):
    # Find indices where emotions match the target emotion ID
    indices = [i for i, emotion in enumerate(ds.emotions) if emotion == target_emotion_id]

    # Extract values for these indices
    emotion_values = [ds.values[i] for i in indices]

    print(f"Found {len(indices)} samples for emotion ID {target_emotion_id}")

    # Return as numpy array for easier manipulation
    return np.array(emotion_values)




data_dir = './EEG_data/'  # Directory containing .mat files

# Create a combined dataset
combined_dataset = None

# Loop through all 20 .mat files
for file_num in range(1, 21):
    file_path = os.path.join(data_dir, f"{file_num}.mat")
    print(f"Processing file: {file_path}")

    # Load the .mat file
    mat_data = DataHelper.load_mat_file(file_path, lds=False)

    # Process data using FFT
    dataset = DataHelper.adapt_to_emotion_format(mat_data)
    emotionDS = EmotionDataset(dataset)
    print(f"Dataset from file {file_num} created with {len(dataset)} samples")

    # Combine datasets
    if combined_dataset is None:
        combined_dataset = emotionDS
    else:
        combined_dataset.extend(emotionDS)

visualizer = Visualizer(100)

visualizer.plot_time_series(
    emotionDS,
    [1,2,3],
    [1,2,3],
    start_sample=0,
    end_sample=5000,
    figsize= (300, 15)
)