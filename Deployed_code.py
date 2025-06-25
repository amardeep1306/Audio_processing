import streamlit as st
import librosa
import numpy as np
import pickle

# --- Load Trained Model and Label Encoder ---
MODEL_PATH = "xgb_model_file.pkl"
ENCODER_PATH = "label_encoder.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# --- Function to Extract MFCC Features from Audio ---
def extract_mfcc_features(audio_file, n_mfcc=40):
    audio, sr = librosa.load(audio_file, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# --- Optional: Emoji Map for Emotions ---
emotion_emoji = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "neutral": "😐",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲"
}

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Speech Emotion Detector", layout="wide", page_icon="🎧")

# --- App Header ---
st.markdown("""
    <h1 style='text-align: center;'>🎧 Voice Emotion Recognition</h1>
    <p style='text-align: center;'>Upload a .wav file and discover the emotion behind the voice using a machine learning model.</p>
""", unsafe_allow_html=True)
st.markdown("---")

# --- Two-Column Layout ---
col1, col2 = st.columns([1, 2])

# --- Left Column: Upload Section ---
with col1:
    st.subheader("📤 Upload Your Audio")
    st.info("Only `.wav` format supported. Please use 16 kHz mono audio.", icon="ℹ️")
    uploaded_file = st.file_uploader("Drop your .wav file below:", type=["wav"])

    if uploaded_file:
        st.success("✅ File received. You can preview below.", icon="📂")
        st.audio(uploaded_file, format="audio/wav")

# --- Right Column: Prediction Section ---
with col2:
    if uploaded_file:
        with st.spinner("🔍 Analyzing and predicting emotion..."):
            try:
                # Extract features and make prediction
                features = extract_mfcc_features(uploaded_file)
                prediction = model.predict([features])
                probabilities = model.predict_proba([features])[0]

                predicted_label = label_encoder.inverse_transform(prediction)[0]
                confidence = float(np.max(probabilities) * 100)

                emoji = emotion_emoji.get(predicted_label.lower(), "🔊")

                # Display Result
                st.success(f"### 🎯 Detected Emotion: {emoji} **{predicted_label.upper()}**")
                

            except Exception as e:
                st.error("❌ Could not process the audio file.")
                st.exception(e)
    else:
        st.warning("Please upload a `.wav` file to begin.", icon="⚠️")

# --- Footer ---
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit | Project by Amardeep</p>", unsafe_allow_html=True)
