import os
import pandas as pd

DATASET_PATH = "data/raw/TESS Toronto emotional speech set data"

data = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(folder_path):

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                parts = file.replace(".wav", "").split("_")

                word = parts[1]

                emotion = parts[2]

                data.append([file_path, word, emotion])

df = pd.DataFrame(
    data,
    columns=["audio_path", "transcript", "emotion"]
)

df.to_csv("metadata.csv", index=False)

print(df.head())

print("\nmetadata.csv created successfully!")