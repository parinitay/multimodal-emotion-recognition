import os
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from transformers import BertTokenizer
from transformers import BertModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

speech_features = []

for path in metadata["audio_path"]:

    feat = extract_features(path)

    speech_features.append(feat)

speech_features = np.array(
    speech_features
)

speech_features = np.transpose(
    speech_features,
    (0,2,1)
)

texts = metadata["transcript"].tolist()

labels = metadata["label"].values

X_train_speech, X_test_speech, train_texts, test_texts, y_train, y_test = train_test_split(

    speech_features,

    texts,

    labels,

    test_size=0.2,

    random_state=42,

    stratify=labels
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

class MultimodalDataset(Dataset):

    def __init__(self, speech, texts, labels):

        self.speech = speech

        self.texts = texts

        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        encoding = tokenizer(

            self.texts[idx],

            padding="max_length",

            truncation=True,

            max_length=16,

            return_tensors="pt"
        )

        return {

            "speech": torch.tensor(

                self.speech[idx],

                dtype=torch.float32
            ),

            "input_ids": encoding["input_ids"].squeeze(),

            "attention_mask": encoding["attention_mask"].squeeze(),

            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

train_dataset = MultimodalDataset(

    X_train_speech,

    train_texts,

    y_train
)

test_dataset = MultimodalDataset(

    X_test_speech,

    test_texts,

    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8
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

class SpeechEncoder(nn.Module):

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

class FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.speech_encoder = SpeechEncoder()

        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.classifier = nn.Sequential(

            nn.Linear(256 + 768, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, 7)
        )

    def forward(

        self,

        speech,

        input_ids,

        attention_mask
    ):

        speech_embedding = self.speech_encoder(
            speech
        )

        text_output = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask
        )

        text_embedding = text_output.last_hidden_state[:,0,:]

        fused = torch.cat(

            [speech_embedding, text_embedding],

            dim=1
        )

        output = self.classifier(fused)

        return output

device = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = FusionModel().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

train_accs = []
test_accs = []

EPOCHS = 5

for epoch in range(EPOCHS):

    model.train()

    train_preds = []
    train_true = []

    total_loss = 0

    for batch in train_loader:

        speech = batch["speech"].to(device)

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(

            speech,

            input_ids,

            attention_mask
        )

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

    train_accs.append(train_acc)

    model.eval()

    test_preds = []
    test_true = []

    with torch.no_grad():

        for batch in test_loader:

            speech = batch["speech"].to(device)

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch["attention_mask"].to(device)

            labels = batch["label"].to(device)

            outputs = model(

                speech,

                input_ids,

                attention_mask
            )

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

    test_accs.append(test_acc)

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

    "models/saved/real_fusion_model.pth"
)

np.save(

    "Results/tables/fusion_train_accuracy.npy",

    np.array(train_accs)
)

np.save(

    "Results/tables/fusion_test_accuracy.npy",

    np.array(test_accs)
)

print("\nREAL multimodal fusion training completed!")