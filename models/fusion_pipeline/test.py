import torch
import torch.nn as nn

class FusionModel(nn.Module):

    def __init__(self):

        super(FusionModel, self).__init__()

        self.speech_fc = nn.Linear(256, 128)

        self.text_fc = nn.Linear(768, 128)

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 7)
        )

    def forward(self, speech_embedding, text_embedding):

        speech_out = self.speech_fc(
            speech_embedding
        )

        text_out = self.text_fc(
            text_embedding
        )

        fused = torch.cat(
            (speech_out, text_out),
            dim=1
        )

        output = self.classifier(fused)

        return output

model = FusionModel()

model.eval()

print("Fusion Model Loaded Successfully!")

dummy_speech_embedding = torch.randn(1, 256)

dummy_text_embedding = torch.randn(1, 768)

with torch.no_grad():

    output = model(
        dummy_speech_embedding,
        dummy_text_embedding
    )

prediction = torch.argmax(
    output,
    dim=1
)

emotion_labels = [

    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

print(
    "Predicted Emotion:",
    emotion_labels[prediction.item()]
)

print("\nFusion Testing Successful!")