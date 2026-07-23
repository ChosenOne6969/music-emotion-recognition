import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
import librosa
import os
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model, scaler, and label encoder
model = xgb.XGBClassifier()
model.load_model(os.path.join(BASE_DIR, 'model_files', 'final_model_xgb.json'))

scaler = joblib.load(os.path.join(BASE_DIR, 'model_files', 'scaler.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'model_files', 'label_encoder.pkl'))

FEATURE_COLS = ['danceability', 'loudness', 'mode', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'tempo',
                 'time_signature', 'key_0', 'key_1', 'key_2', 'key_3',
                 'key_4', 'key_5', 'key_6', 'key_7', 'key_8', 'key_9',
                 'key_10', 'key_11']

EMOTION_INFO = {
    'Happy':  {'color': '#f5b700', 'desc': 'High valence, high energy — upbeat and joyful.'},
    'Angry':  {'color': '#e63946', 'desc': 'Low valence, high energy — intense and tense.'},
    'Calm':   {'color': '#4ea8de', 'desc': 'High valence, low energy — peaceful and content.'},
    'Sad':    {'color': '#5c6784', 'desc': 'Low valence, low energy — melancholic and subdued.'},
}

# Dataset medians used as defaults for features that can't be extracted from raw audio
DEFAULT_FEATURES = {
    'danceability': 0.55, 'mode': 1, 'speechiness': 0.05,
    'acousticness': 0.30, 'instrumentalness': 0.0, 'liveness': 0.12,
    'time_signature': 4, 'key': 0
}


def home(request):
    return render(request, 'predictor/home.html')


def about(request):
    return render(request, 'predictor/about.html')


def predict_emotion(request):
    result = None
    confidence = None
    emotion_color = None
    emotion_desc = None
    probabilities = None

    if request.method == 'POST':
        danceability = float(request.POST.get('danceability'))
        loudness = float(request.POST.get('loudness'))
        mode = int(request.POST.get('mode'))
        speechiness = float(request.POST.get('speechiness'))
        acousticness = float(request.POST.get('acousticness'))
        instrumentalness = float(request.POST.get('instrumentalness'))
        liveness = float(request.POST.get('liveness'))
        tempo = float(request.POST.get('tempo'))
        time_signature = int(request.POST.get('time_signature'))
        key = int(request.POST.get('key'))

        row = {col: 0 for col in FEATURE_COLS}
        row['danceability'] = danceability
        row['loudness'] = loudness
        row['mode'] = mode
        row['speechiness'] = speechiness
        row['acousticness'] = acousticness
        row['instrumentalness'] = instrumentalness
        row['liveness'] = liveness
        row['tempo'] = tempo
        row['time_signature'] = time_signature
        row[f'key_{key}'] = 1

        input_df = pd.DataFrame([row])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        result = le.inverse_transform([pred])[0]
        confidence = round(max(proba) * 100, 1)
        emotion_color = EMOTION_INFO[result]['color']
        emotion_desc = EMOTION_INFO[result]['desc']

        probabilities = sorted(
            [(le.inverse_transform([i])[0], round(p * 100, 1)) for i, p in enumerate(proba)],
            key=lambda x: -x[1]
        )

    return render(request, 'predictor/predict.html', {
        'result': result,
        'confidence': confidence,
        'emotion_color': emotion_color,
        'emotion_desc': emotion_desc,
        'probabilities': probabilities,
    })


def upload_predict(request):
    result = None
    confidence = None
    emotion_color = None
    emotion_desc = None
    probabilities = None
    extracted = None

    if request.method == 'POST' and request.FILES.get('audio_file'):
        audio_file = request.FILES['audio_file']

        y, sr = librosa.load(audio_file, sr=None, duration=30)  # analyze first 30 sec

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo)

        rms = librosa.feature.rms(y=y)[0]
        loudness = float(20 * np.log10(np.mean(rms) + 1e-6))  # approximate dB

        extracted = {'tempo': round(tempo, 1), 'loudness': round(loudness, 1)}

        row = {col: 0 for col in FEATURE_COLS}
        row['tempo'] = tempo
        row['loudness'] = loudness
        row['danceability'] = DEFAULT_FEATURES['danceability']
        row['mode'] = DEFAULT_FEATURES['mode']
        row['speechiness'] = DEFAULT_FEATURES['speechiness']
        row['acousticness'] = DEFAULT_FEATURES['acousticness']
        row['instrumentalness'] = DEFAULT_FEATURES['instrumentalness']
        row['liveness'] = DEFAULT_FEATURES['liveness']
        row['time_signature'] = DEFAULT_FEATURES['time_signature']
        row[f"key_{DEFAULT_FEATURES['key']}"] = 1

        input_df = pd.DataFrame([row])[FEATURE_COLS]
        input_scaled = scaler.transform(input_df)

        pred = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        result = le.inverse_transform([pred])[0]
        confidence = round(max(proba) * 100, 1)
        emotion_color = EMOTION_INFO[result]['color']
        emotion_desc = EMOTION_INFO[result]['desc']
        probabilities = sorted(
            [(le.inverse_transform([i])[0], round(p * 100, 1)) for i, p in enumerate(proba)],
            key=lambda x: -x[1]
        )

    return render(request, 'predictor/upload.html', {
        'result': result,
        'confidence': confidence,
        'emotion_color': emotion_color,
        'emotion_desc': emotion_desc,
        'probabilities': probabilities,
        'extracted': extracted,
    })