import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Emotion Recognition AI",
    page_icon="💗",
    layout="wide"
)

# ---------------- FORCE LIGHT MODE ---------------- #

st.markdown("""
<meta name="color-scheme" content="light">
""", unsafe_allow_html=True)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>

/* ---------------- GLOBAL ---------------- */

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

html {
    color-scheme: light !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #FFF7FA, #FFFDFD);
}

.block-container {
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* ---------------- SIDEBAR ---------------- */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #B31952 0%, #920F3F 100%);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem;
}

/* ---------------- SIDEBAR CONTENT ---------------- */

.sidebar-title {
    color: white !important;
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0;
}

.sidebar-sub {
    color: #FFE7EF !important;
    text-align: center;
    font-size: 15px;
    margin-bottom: 40px;
}

.nav-card {
    background: rgba(255,255,255,0.13);
    padding: 18px;
    border-radius: 18px;
    margin-bottom: 16px;
    color: white !important;
    font-size: 20px;
    font-weight: 600;
    transition: 0.3s;
}

.nav-card:hover {
    background: #E6F082;
    color: #8B1240 !important;
    transform: translateX(4px);
}

/* ---------------- TITLE ---------------- */

.main-title {
    text-align: center;
    font-size: 76px;
    font-weight: 800;
    color: #B31952 !important;
    line-height: 1;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #6D5A62 !important;
    margin-bottom: 55px;
}

/* ---------------- LABELS ---------------- */

.section-label {
    text-align: center;
    color: #B31952 !important;
    font-size: 17px;
    letter-spacing: 4px;
    font-weight: 700;
    margin-bottom: 15px;
}

/* ---------------- RADIO BUTTONS ---------------- */

.stRadio > div {
    justify-content: center;
    gap: 20px;
}

.stRadio label {
    background: white !important;
    border: 2px solid #F5D2DE !important;
    border-radius: 18px !important;
    padding: 14px 30px !important;
    font-weight: 600 !important;
    color: #8A2348 !important;
    transition: 0.3s;
}

.stRadio label:hover {
    border: 2px solid #E6F082 !important;
    background: #FFFDF2 !important;
}

.stRadio label span {
    color: #8A2348 !important;
}

/* ---------------- FILE UPLOADER ---------------- */

[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #F5B5C7 !important;
    border-radius: 28px !important;
    padding: 30px !important;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.05);
}

[data-testid="stFileUploader"] section {
    background: white !important;
}

[data-testid="stFileUploader"] * {
    color: #7A1D3D !important;
}

/* ---------------- AUDIO INPUT ---------------- */

[data-testid="stAudioInput"] * {
    color: #7A1D3D !important;
}

/* ---------------- BUTTONS ---------------- */

.stButton > button {
    background: linear-gradient(90deg, #E6F082, #D6ED67);
    color: #8A1C45 !important;
    border: none;
    border-radius: 16px;
    padding: 0.9rem 2rem;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s;
    box-shadow: 0px 8px 24px rgba(230,240,130,0.35);
}

.stButton > button:hover {
    transform: scale(1.03);
}

/* ---------------- RESULT CARD ---------------- */

.result-card {
    background: white;
    padding: 40px;
    border-radius: 30px;
    box-shadow: 0px 10px 35px rgba(0,0,0,0.05);
    margin-top: 30px;
    border: 1px solid #F5D7E2;
}

.result-heading {
    text-align: center;
    color: #B31952 !important;
    font-size: 17px;
    letter-spacing: 3px;
    font-weight: 700;
}

.result-emotion {
    text-align: center;
    color: #B31952 !important;
    font-size: 54px;
    font-weight: 800;
    margin-top: 15px;
}

.result-confidence {
    text-align: center;
    color: #7A5D68 !important;
    font-size: 22px;
    margin-top: 10px;
}

/* ---------------- PROGRESS BAR ---------------- */

.stProgress > div > div > div > div {
    background: linear-gradient(to right, #E6F082, #D4ED5A);
}

/* ---------------- AUDIO PLAYER ---------------- */

audio {
    width: 100%;
    border-radius: 18px;
    margin-top: 10px;
}

/* ---------------- WAVEFORM CARD ---------------- */

.wave-card {
    background: white;
    border-radius: 28px;
    padding: 25px;
    margin-top: 30px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.05);
}

/* ---------------- FOOTER ---------------- */

.footer {
    text-align: center;
    color: #8B3354 !important;
    margin-top: 40px;
    opacity: 0.85;
}

/* ---------------- FORCE TEXT VISIBILITY ---------------- */

html, body, p, span, label, div {
    color: #7A1D3D !important;
}

label {
    color: #8B2348 !important;
}

.stRadio label,
.stRadio div,
.stRadio span {
    color: #8B2348 !important;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div {
    color: #7A1D3D !important;
}

.stButton button {
    color: #8A1C45 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: white !important;
}

/* FIX BRAVE DARK MODE */
* {
    -webkit-text-fill-color: unset !important;
}

/* ---------------- REMOVE STREAMLIT DEFAULTS ---------------- */

header {
    background: transparent !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #

st.sidebar.markdown("""

<div class="sidebar-title">
💗 Emotion AI
</div>

<div class="sidebar-sub">
Understand Emotions.<br>
Build Connections.
</div>

<div class="nav-card">🏠 Home</div>
<div class="nav-card">🎤 Emotion Detection</div>
<div class="nav-card">📊 Visualizations</div>
<div class="nav-card">ℹ️ About Project</div>

<br><br><br>

<div style='text-align:center;'>

<h3 style='color:white;'>
✨ AI that listens.
</h3>

<p style='color:#FFE7EF;'>
Emotions that matter.
</p>

</div>

""", unsafe_allow_html=True)

# ---------------- TITLE ---------------- #

st.markdown("""
<div class="main-title">
Emotion Recognition System
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
Understanding human emotions through speech using Deep Learning
</div>
""", unsafe_allow_html=True)

# ---------------- EMOTION LABELS ---------------- #

emotion_labels = [
    "Angry 😠",
    "Disgust 😖",
    "Fear 😨",
    "Happy 😄",
    "Neutral 😐",
    "Pleasant Surprise 😲",
    "Sad 😢"
]

# ---------------- FEATURE EXTRACTION ---------------- #

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

# ---------------- ATTENTION ---------------- #

class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(hidden_dim * 2, 1)

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

# ---------------- MODEL ---------------- #

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

        self.fc = nn.Linear(256, 7)

    def forward(self, x):

        x = x.permute(0,2,1)

        x = self.cnn(x)

        x = x.permute(0,2,1)

        lstm_out, _ = self.lstm(x)

        context = self.attention(lstm_out)

        context = self.dropout(context)

        output = self.fc(context)

        return output

# ---------------- LOAD MODEL ---------------- #

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

model = SpeechEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "models/saved/speech_model.pth",
        map_location=device
    )
)

model.eval()

# ---------------- INPUT METHOD ---------------- #

st.markdown("""
<div class="section-label">
CHOOSE INPUT METHOD
</div>
""", unsafe_allow_html=True)

option = st.radio(
    "",
    ["Upload Audio", "Record Audio"],
    horizontal=True
)

audio_path = None

# ---------------- FILE UPLOAD ---------------- #

if option == "Upload Audio":

    uploaded_file = st.file_uploader(
        "Upload WAV File",
        type=["wav"]
    )

    if uploaded_file is not None:

        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.read())

        audio_path = "temp.wav"

        st.audio(audio_path)

# ---------------- AUDIO RECORD ---------------- #

if option == "Record Audio":

    audio_bytes = st.audio_input(
        "Record your voice"
    )

    if audio_bytes is not None:

        with open("recorded.wav", "wb") as f:
            f.write(audio_bytes.read())

        audio_path = "recorded.wav"

        st.audio(audio_path)

# ---------------- PREDICTION ---------------- #

if audio_path is not None:

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("✨ Predict Emotion"):

        with st.spinner("Analyzing Emotion..."):

            features = extract_features(audio_path)

            features = np.transpose(features)

            tensor = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():

                outputs = model(tensor)

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                prediction = torch.argmax(
                    probs,
                    dim=1
                ).item()

            confidence = probs[0][prediction].item()

        # ---------------- RESULT CARD ---------------- #

        st.markdown(f"""

        <div class="result-card">

            <div class="result-heading">
            PREDICTION RESULT
            </div>

            <div class="result-emotion">
            {emotion_labels[prediction]}
            </div>

            <div class="result-confidence">
            Confidence Score: {confidence*100:.2f}%
            </div>

        </div>

        """, unsafe_allow_html=True)

        st.progress(float(confidence))

        # ---------------- WAVEFORM ---------------- #

        audio, sr = librosa.load(
            audio_path,
            sr=16000
        )

        st.markdown("""
        <div class="wave-card">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="section-label" style="text-align:left;">
        AUDIO WAVEFORM
        </div>
        """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(12,3))

        ax.plot(
            audio,
            linewidth=1.8,
            color="#B31952"
        )

        ax.set_facecolor("#FFF9FC")

        fig.patch.set_facecolor("#FFFFFF")

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_color("#E7B5C7")
        ax.spines['bottom'].set_color("#E7B5C7")

        ax.tick_params(colors="#8B3354")

        ax.set_xlabel("Time", color="#8B3354")
        ax.set_ylabel("Amplitude", color="#8B3354")

        plt.tight_layout()

        st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ---------------- #

st.markdown("""

<div class="footer">

<hr style="border:1px solid #F3D4DF;">

Built with 💗 using CNN • BiLSTM • Attention • Streamlit

</div>

""", unsafe_allow_html=True)