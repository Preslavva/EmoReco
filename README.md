This project implements a **Speech Emotion Recognition (SER)** system that predicts emotions from speech signals using deep learning and interpretable analysis techniques. The model processes audio recordings, extracts MFFCS and  **log-Mel spectrogram features**, and learns emotional patterns using baseline machine learning models (kNN, Random Forest and SVM) and  **Convolutional Neural Network (CNN) combined with a Bidirectional LSTM (BiLSTM)** architecture. I further experimented with pretrained models and specifically with their embeddings (WavLM and Wav2Vec) which I fed into MLP (**Multi-layer Perceptron)**

## Features

- Exploratory Data Analysis (EDA)
- Audio preprocessing pipeline for speech data
- MFFCs extraction
- Log-Mel spectrogram feature extraction
- kNN, Random Forest and SVM training
- CNN–BiLSTM deep learning architecture for emotion classification
- Wav2Vec and WavLm embeddings + MLP/SVM
- Evaluation using **accuracy and macro F1-score**
- Visualization of model performance using **confusion matrices**
- Explainable AI using **Grad-CAM** and **time-importance analysis**
- Simple **Flask web interface** for uploading audio and predicting emotions

## Dataset

The model is trained and evaluated on the **CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)**, which contains acted emotional speech recordings labeled with six emotions:

- Anger
- Disgust
- Fear
- Happiness
- Neutral
- Sadness

## Explainable AI

To improve transparency and trust in the model, the project includes:

- **Grad-CAM visualizations** to highlight important time–frequency regions
- **Time-importance curves** showing which moments in speech influence predictions
- **Confusion matrix analysis** to understand model behavior and limitations

## Applications

Speech emotion recognition can support applications such as:

- Human–computer interaction
- Customer service analytics
- Mental health monitoring
- Assistive technologies
- Emotion-aware virtual assistants

## Technologies Used

- Python
- PyTorch
- Librosa
- NumPy
- Matplotlib
- Flask
