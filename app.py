from flask import Flask, render_template, request
import os
import joblib
import numpy as np
import librosa

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model and scaler
model = joblib.load('genre_classification_model.pkl')  # Load trained classifier
scaler = joblib.load('scaler.pkl')  # Load feature scaler
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


    # Extract Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

    # Extract Chroma Features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma.T, axis=0)

    # Combine all features
    return np.hstack([mfccs_mean, spectral_contrast_mean, chroma_mean])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Extract features
        features = extract_features(filepath)
        features = features.reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict genre using trained model
        prediction = model.predict(features_scaled)
        predicted_genre = GENRES[int(prediction[0])]

        return render_template('result.html', genre=predicted_genre, filename=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
