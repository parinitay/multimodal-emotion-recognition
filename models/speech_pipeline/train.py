import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

metadata = pd.read_csv("metadata.csv")

encoder = LabelEncoder()

metadata["label"] = encoder.fit_transform(
    metadata["emotion"]
)

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

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

class SpeechDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]

train_dataset = SpeechDataset(
    X_train,
    y_train
)

test_dataset = SpeechDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

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

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            256,
            7
        )

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.cnn(x)

        x = x.permute(0,2,1)

        lstm_out, _ = self.lstm(x)

        context = self.attention(
            lstm_out
        )

        context = self.dropout(context)

        output = self.fc(context)

        return output

device = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = SpeechEmotionModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

train_accuracies = []
test_accuracies = []

EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()

    train_preds = []
    train_true = []

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            outputs,
            dim=1
        )

        train_preds.extend(
            preds.cpu().numpy()
        )

        train_true.extend(
            labels.cpu().numpy()
        )

    train_acc = accuracy_score(
        train_true,
        train_preds
    )

    train_accuracies.append(
        train_acc
    )

    model.eval()

    test_preds = []
    test_true = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            test_preds.extend(
                preds.cpu().numpy()
            )

            test_true.extend(
                labels.cpu().numpy()
            )

    test_acc = accuracy_score(
        test_true,
        test_preds
    )

    test_accuracies.append(
        test_acc
    )

    print(

        f"Epoch {epoch+1}/{EPOCHS} | "

        f"Loss: {total_loss:.4f} | "

        f"Train Acc: {train_acc:.4f} | "

        f"Test Acc: {test_acc:.4f}"
    )

os.makedirs(
    "models/saved",
    exist_ok=True
)

torch.save(

    model.state_dict(),

    "models/saved/speech_model.pth"
)

np.save(

    "Results/tables/speech_train_accuracy.npy",

    np.array(train_accuracies)
)

np.save(

    "Results/tables/speech_test_accuracy.npy",

    np.array(test_accuracies)
)

print("\nREAL speech training completed!")