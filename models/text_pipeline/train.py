import pandas as pd
import torch
import numpy as np

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

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

train_texts, test_texts, train_labels, test_labels = train_test_split(

    metadata["transcript"].tolist(),

    metadata["label"].tolist(),

    test_size=0.2,

    random_state=42
)

tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

class TextDataset(Dataset):

    def __init__(self, texts, labels):

        self.texts = texts
        self.labels = labels

    def __len__(self):

        return len(self.texts)

    def __getitem__(self, idx):

        encoding = tokenizer(

            self.texts[idx],

            padding="max_length",

            truncation=True,

            max_length=16,

            return_tensors="pt"
        )

        return {

            "input_ids": encoding["input_ids"].squeeze(),

            "attention_mask": encoding["attention_mask"].squeeze(),

            "label": torch.tensor(
                self.labels[idx],
                dtype=torch.long
            )
        }

train_dataset = TextDataset(
    train_texts,
    train_labels
)

test_dataset = TextDataset(
    test_texts,
    test_labels
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

model = BertForSequenceClassification.from_pretrained(

    "bert-base-uncased",

    num_labels=7
)

device = torch.device(

    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model.to(device)

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=2e-5
)

train_accuracies = []

EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()

    predictions = []
    true_labels = []

    total_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch["attention_mask"].to(device)

        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(

            input_ids=input_ids,

            attention_mask=attention_mask,

            labels=labels
        )

        loss = outputs.loss

        logits = outputs.logits

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(
            logits,
            dim=1
        )

        predictions.extend(
            preds.cpu().numpy()
        )

        true_labels.extend(
            labels.cpu().numpy()
        )

    acc = accuracy_score(
        true_labels,
        predictions
    )

    train_accuracies.append(acc)

    print(

        f"Epoch {epoch+1}/{EPOCHS} | "

        f"Loss: {total_loss:.4f} | "

        f"Accuracy: {acc:.4f}"
    )

torch.save(

    model.state_dict(),

    "models/saved/text_model.pth"
)

np.save(

    "Results/tables/text_accuracy.npy",

    np.array(train_accuracies)
)

print("\nText model training completed!")