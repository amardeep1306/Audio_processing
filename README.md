#  Emotion Classification from Audio (Speech/Song)

This project implements a complete pipeline to classify **emotions from speech or song audio files** using **machine learning** techniques. The application allows users to upload `.wav` files and get real-time emotion predictions through a simple and interactive web interface built with **Streamlit**.

---

##  Objective

The goal is to design a robust and accurate end-to-end system that can:
- Extract emotional features from speech/song audio.
- Classify the audio into one of several emotional categories.
- Deploy the model as a user-friendly web application.

---

##  Features

-  **High Accuracy**: F1 Score > 80%, Class-wise accuracy > 75%
-  **Supports .wav Audio**: Designed to process high-quality `.wav` files
-  **MFCC-Based Feature Extraction**: Captures emotion-relevant patterns from speech
-  **Trained using XGBoost**: Efficient and accurate classification
-  **Streamlit App**: Easy-to-use browser-based interface
-  **Ready for Deployment**

---

##  Dataset

The dataset consists of `.wav` audio files categorized by emotions. 
>  Speech and song audio types are included.  
>  File names to download are shared separately in the doubt group.

---

## Installation and Running Instructions

### Requirements

Install dependencies using:

```bash
pip install streamlit librosa scikit-learn xgboost numpy
```
## Run the Web App

To start the Streamlit app:

```bash
streamlit run Deployed_code.py
```
##  Usage

- A web interface will open in your browser.
- Upload a `.wav` audio file.
- The app will extract features and display the predicted emotion.

---

##  Model Details

- **Model**: `XGBoostClassifier`
- **Input Features**: 40-dimensional MFCCs
- **Trained On**: Speech and song datasets
- **Saved Model File**: `xgb_model_file.pkl`
- **Label Encoder**: `label_encoder.pkl`

---

## Project Structure

├── Deployed_code.py 

├── xgb_model_file.pkl 

├── label_encoder.pkl  

├── audio.ipynb  

└── README.md                 

##  Future Work

-  **Real-time audio streaming support**
-  **Integration of CNN/RNN for deep feature learning**
-  **Multilingual emotional audio classification**
-  **Add waveform and spectrogram visualizations**

---

##  Authors

- [Amardeep Kumar](https://github.com/amardeep1306) *(Project Developer)*

---
