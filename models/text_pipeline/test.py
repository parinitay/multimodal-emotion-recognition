from transformers import BertTokenizer
from transformers import BertForSequenceClassification

import torch

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=7
)

model.eval()

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

sample_sentences = [

    "I am extremely angry right now",

    "This is the best day of my life",

    "I feel very sad and lonely",

    "Wow I did not expect this surprise",

    "I am scared of what will happen",

    "Everything feels normal today"
]

for sentence in sample_sentences:

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    print("\nSentence:", sentence)

    print(
        "Predicted Emotion:",
        emotion_labels[prediction]
    )