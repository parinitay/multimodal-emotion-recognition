import torch
import torch.nn as nn
import numpy as np
import librosa

class BiLSTMModel(nn.Module):

    def __init__(self):

        super(BiLSTMModel, self).__init__()

        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 7)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)

        output = lstm_out[:, -1, :]

        output = self.fc(output)

        return output

model = BiLSTMModel()

model.load_state_dict(
    torch.load("models/saved/speech_model.pth")
)

model.eval()

print("Speech model loaded successfully!")

sample_audio = "data/raw/TESS Toronto emotional speech set data/OAF_angry/OAF_back_angry.wav"

audio, sample_rate = librosa.load(
    sample_audio,
    sr=16000
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

max_pad_len = 200

pad_width = max_pad_len - mfcc.shape[1]

if pad_width > 0:

    mfcc = np.pad(
        mfcc,
        pad_width=((0, 0), (0, pad_width)),
        mode='constant'
    )

else:

    mfcc = mfcc[:, :max_pad_len]

mfcc = np.transpose(mfcc, (1, 0))

mfcc_tensor = torch.tensor(
    mfcc,
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():

    output = model(mfcc_tensor)

    prediction = torch.argmax(output, dim=1)

print("Predicted Emotion Class:", prediction.item())