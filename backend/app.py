import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app) # Allow React to connect

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Models (Ensure .h5 files are in backend/models/)
try:
    print("Loading models...")
    image_model = tf.keras.models.load_model('models/deepfake_image_model.h5')
    audio_model = tf.keras.models.load_model('models/deepfake_audio_model.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

# Helper: Prepare Image
def prepare_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128)) # Must match training input
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Helper: Prepare Audio
def prepare_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050, duration=3)
    if len(y) < 22050 * 3:
        y = np.pad(y, (0, 22050 * 3 - len(y)))
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel[np.newaxis, ..., np.newaxis] # Add batch & channel dims

# --- Endpoints ---

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    
    # Predict
    img = prepare_image(path)
    pred = image_model.predict(img)[0][0]
    result = "REAL" if pred > 0.5 else "FAKE" # Adjust based on training labels
    
    return jsonify({'result': result, 'confidence': float(pred)})

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    
    # Process Video: Sample 10 frames
    cap = cv2.VideoCapture(path)
    predictions = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Process every 30th frame
        if frame_count % 30 == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)) / 255.0
            pred = image_model.predict(np.expand_dims(frame, axis=0), verbose=0)[0][0]
            predictions.append(pred)
        frame_count += 1
    
    cap.release()
    
    if not predictions: return jsonify({'error': 'Could not process video'}), 500
    
    avg_pred = np.mean(predictions)
    result = "REAL" if avg_pred > 0.5 else "FAKE"
    return jsonify({'result': result, 'confidence': float(avg_pred)})

@app.route('/detect-audio', methods=['POST'])
def detect_audio():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(path)
    
    features = prepare_audio(path)
    pred = audio_model.predict(features)[0][0]
    result = "REAL" if pred > 0.5 else "FAKE" # Check your training labels!
    
    return jsonify({'result': result, 'confidence': float(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)