import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

metadata = pd.read_csv("metadata.csv")

encoder = LabelEncoder()

metadata["label"] = encoder.fit_transform(
    metadata["emotion"]
)

emotion_names = encoder.classes_

def extract_features(file_path, max_pad_len=200):

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000
    )

    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    combined = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    pad_width = max_pad_len - combined.shape[1]

    if pad_width > 0:

        combined = np.pad(

            combined,

            pad_width=((0,0),(0,pad_width)),

            mode='constant'
        )

    else:

        combined = combined[:, :max_pad_len]

    return combined

features = []

for path in metadata["audio_path"]:

    feat = extract_features(path)

    features.append(feat)

X = np.array(features)

X = np.transpose(X, (0,2,1))

y = metadata["label"].values

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, lstm_output):

        weights = torch.softmax(

            self.attention(lstm_output),

            dim=1
        )

        context = torch.sum(
            weights * lstm_output,
            dim=1
        )

        return context

class SpeechEmotionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                120,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=128,

            batch_first=True,

            bidirectional=True
        )

        self.attention = Attention(128)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.cnn(x)

        x = x.permute(0,2,1)

        lstm_out, _ = self.lstm(x)

        context = self.attention(
            lstm_out
        )

        return context

device = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = SpeechEmotionModel().to(device)

saved_weights = torch.load(

    "models/saved/speech_model.pth",

    map_location=device
)

filtered_weights = {

    k:v for k,v in saved_weights.items()

    if "fc" not in k
}

model.load_state_dict(
    filtered_weights,
    strict=False
)

model.eval()

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
).to(device)

with torch.no_grad():

    embeddings = model(X_tensor)

embeddings = embeddings.cpu().numpy()

tsne = TSNE(

    n_components=2,

    random_state=42
)

reduced = tsne.fit_transform(
    embeddings
)

plt.figure(figsize=(10,8))

for idx, emotion in enumerate(emotion_names):

    indices = y == idx

    plt.scatter(

        reduced[indices, 0],

        reduced[indices, 1],

        label=emotion
    )

plt.legend()

plt.title("REAL t-SNE Emotion Embeddings")

plt.xlabel("Dimension 1")

plt.ylabel("Dimension 2")

plt.savefig(

    "Results/embeddings/real_tsne.png"
)

plt.show()

print("\nREAL t-SNE visualization generated!")