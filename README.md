# 🎙️ EmoSense — Multimodal Emotion Recognition using CNN–BiLSTM–Attention and BERT

EmoSense is an AI-powered Multimodal Emotion Recognition system that predicts human emotions from speech and text inputs.

The system combines acoustic speech modelling and contextual language understanding through deep learning to classify emotions into predefined categories.

The project uses the Toronto Emotional Speech Set (TESS) dataset and includes:

-  Speech Emotion Recognition
-  Text Emotion Recognition
-  Multimodal Fusion
-  Embedding Visualisation
-  Interactive Streamlit Application

---

# 🗂️ Project Structure

```bash
📦 project/

├── 📂 data/
│   └── 📂 raw/
│
├── 📂 models/
│   │
│   ├── 📂 speech_pipeline/
│   │   ├── 📄 train.py
│   │   ├── 📄 test.py
│   │
│   ├── 📂 text_pipeline/
│   │   ├── 📄 train.py
│   │   ├── 📄 test.py
│   │   └── 📄 new_test.py
│   │
│   └── 📂 fusion_pipeline/
│       ├── 📄 train.py
│       ├── 📄 test.py
│       └── 📄 new_test.py
│
├── 📂 Results/
│   ├── 📂 plots/
│   ├── 📂 embeddings/
│   └── 📂 confusion_matrices/
│
├── 📂 models/saved/
│
├── 📄 app.py
├── 📄 create_metadata.py
├── 📄 real_tsne.py
├── 📄 real_confusion_matrix.py
├── 📄 model_comparison.py
├── 📄 results_visualization.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

# 📦 Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone YOUR_GITHUB_LINK
```

Move into folder:

```bash
cd project
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Main Dependencies

## Deep Learning

```bash
torch
transformers
scikit-learn
numpy
```

## Audio Processing

```bash
librosa
soundfile
soxr
```

## Data Processing

```bash
pandas
matplotlib
```

## UI

```bash
streamlit
```

---

# 📊 Dataset

Dataset Used:

Toronto Emotional Speech Set (TESS)

Contains:

- Speech samples
- Text transcripts
- Emotion labels

Emotion Classes:

```text
Angry
Disgust
Fear
Happy
Neutral
Pleasant Surprise
Sad
```

Store dataset inside:

```bash
data/raw/
```

---

# ⚙️ Generate Metadata

```bash
python create_metadata.py
```

Generates:

```text
metadata.csv
```

---

# 🎤 Speech Pipeline

Train:

```bash
python models/speech_pipeline/train.py
```

Test:

```bash
python models/speech_pipeline/test.py
```

Output:

```text
Speech Accuracy
Speech Confusion Matrix
Speech Embeddings
```

---

# 📝 Text Pipeline

Train:

```bash
python models/text_pipeline/train.py
```

Generate Evaluation:

```bash
python models/text_pipeline/new_test.py
```

Output:

```text
Text Accuracy
Text Confusion Matrix
Text Embeddings
```

---

# 🔀 Fusion Pipeline

Train:

```bash
python models/fusion_pipeline/train.py
```

Generate Evaluation:

```bash
python models/fusion_pipeline/new_test.py
```

Output:

```text
Fusion Accuracy
Fusion Confusion Matrix
Fusion Embeddings
```

---

# 📈 Generate Visualisations

Accuracy Curve:

```bash
python results_visualization.py
```

Confusion Matrix:

```bash
python real_confusion_matrix.py
```

t-SNE:

```bash
python real_tsne.py
```

Model Comparison:

```bash
python model_comparison.py
```

---

# 🌸 Run Streamlit Interface

Launch Application:

```bash
streamlit run app.py
```

Features:

- Upload Audio
- Record Audio
- Predict Emotion
- Confidence Score
- Waveform Visualisation

---

# 🧠 Architecture Summary

| Block | Architecture |
|-------|-------------|
| Speech Feature Extraction | MFCC + Delta + Delta² |
| Temporal Modelling | CNN + BiLSTM + Attention |
| Text Feature Extraction | BERT |
| Fusion | Feature Concatenation |
| Classification | Dense Neural Network |

---

# 📊 Results Summary

| Model | Performance |
|-------|-------------|
| Speech-only | High |
| Text-only | Moderate |
| Fusion | Highest |

Generated Outputs:

- Accuracy Curves
- Confusion Matrix
- t-SNE Embeddings
- Fusion Visualisations

---

# 🔬 Key Observations

- Speech contributed more strongly in TESS.
- Text performance was limited by transcript characteristics.
- Fusion improved representation richness.

---

# 🖥️ Interface

The project includes a custom Streamlit application with:

- Upload Interface
- Recording Interface
- Emotion Prediction
- Visual Feedback

---

# 🛠 Tech Stack

```text
Python
PyTorch
Transformers
Librosa
Scikit-Learn
Streamlit
Matplotlib
```

---

# 👩‍💻 Author

Developed as part of a Deep Learning based Multimodal Emotion Recognition Project  - by parinita


OUTPUT:

<img width="1918" height="916" alt="Screenshot 2026-05-20 141516" src="https://github.com/user-attachments/assets/24d91b1e-dcf0-4fd5-b751-1dce07680b60" />

<img width="1918" height="916" alt="Screenshot 2026-05-20 141516" src="https://github.com/user-attachments/assets/b86c539c-cd98-4f09-82f6-ee20f0d2e02e" />

<img width="1918" height="966" alt="Screenshot 2026-05-20 141419" src="https://github.com/user-attachments/assets/d7e9c298-7027-41a3-a08e-6706a049e346" />

<img width="1918" height="916" alt="Screenshot 2026-05-20 141516" src="https://github.com/user-attachments/assets/bbc35197-b974-4412-befc-ff8b58ec7cd0" />

<img width="1918" height="1020" alt="Screenshot 2026-05-20 141406" src="https://github.com/user-attachments/assets/52c00411-b300-4c8f-a3c9-d1c43e1ea225" />









