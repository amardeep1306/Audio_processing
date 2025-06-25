import streamlit as st
import librosa
import numpy as np
import pickle
import os

# Load trained model and label encoder
model_path = "xgb_model_audio.pkl"
encoder_path = "label_encoder_audio.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Feature extraction
def extract_mfcc(wav_file, n_mfcc=40):
    y, sr = librosa.load(wav_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# Streamlit UI
st.title("Emotion Classification from Audio")
st.write("Upload a `.wav` file and the model will predict the emotion expressed in the voice.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    with st.spinner("Analyzing audio..."):
        try:
            features = extract_mfcc(uploaded_file)
            prediction = model.predict([features])
            emotion = label_encoder.inverse_transform(prediction)[0]

            st.success(f" Detected Emotion: **{emotion.upper()}**")
        except Exception as e:
            st.error(f"Error while processing the audio: {e}")
