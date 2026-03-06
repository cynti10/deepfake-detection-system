import os
# Ensure standalone Keras targets TensorFlow backend
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import cv2
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading
import uuid
import time

# Thread lock for model access
model_lock = threading.Lock()

CWD = os.getcwd()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app) 

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', str(50 * 1024 * 1024)))

# Global Models
image_model = None
audio_model = None
model_load_errors = {}

# --- MODEL LOADING LOGIC ---
def load_keras_model(path):
    """Robustly loads a Keras model (.keras or .h5)."""
    try:
        print(f"Loading model from: {path}")
        # compile=False allows loading even if optimizer classes changed slightly
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        error_msg = f"Error loading {os.path.basename(path)}: {str(e)}"
        print(error_msg)
        model_load_errors[os.path.basename(path)] = str(e)
        return None

print("--- INITIALIZING MODELS ---")

# 1. LOAD IMAGE MODEL (Prioritize the new .keras MobileNetV2)
image_paths = [
    os.path.join(BASE_DIR, 'models', 'deepfake_image_model.keras'), # Best format
    os.path.join(BASE_DIR, 'models', 'deepfake_image_model.h5'),    # Legacy
]

for ipath in image_paths:
    if os.path.exists(ipath):
        image_model = load_keras_model(ipath)
        if image_model:
            print(f"✅ Image Model Loaded: {os.path.basename(ipath)}")
            break

if image_model is None:
    print("❌ CRITICAL: No Image Model loaded. Please train the MobileNetV2 model and place 'deepfake_image_model.keras' in /models.")

# 2. LOAD AUDIO MODEL
audio_paths = [
    os.path.join(BASE_DIR, 'models', 'deepfake_audio_model.keras'),
    os.path.join(BASE_DIR, 'models', 'deepfake_audio_model.h5'),
]

for apath in audio_paths:
    if os.path.exists(apath):
        audio_model = load_keras_model(apath)
        if audio_model:
            print(f"✅ Audio Model Loaded: {os.path.basename(apath)}")
            break

if audio_model is None:
    print("⚠️ Warning: No Audio Model loaded.")

# --- HELPER FUNCTIONS ---

def prepare_image(image_path):
    """Prepares image for MobileNetV2 (224x224)"""
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # RESIZE TO 224x224 (Match the training code!)
    img = cv2.resize(img, (224, 224)) 
    
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def prepare_audio(audio_path):
    """Prepares audio spectrogram"""
    try:
        librosa.cache.clear()
        y, sr = librosa.load(audio_path, sr=22050, duration=3)
        
        target_len = 22050 * 3
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        elif len(y) > target_len:
            y = y[:target_len]
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel[np.newaxis, ..., np.newaxis]
    except Exception as e:
        print(f"Audio Prep Error: {e}")
        raise

# --- ENDPOINTS ---

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if not image_model:
        return jsonify({'error': 'Image model not active. Check server logs.'}), 503
    if 'file' not in request.files: 
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    # Use unique filename to avoid Windows file locking
    file_ext = os.path.splitext(secure_filename(file.filename))[1] or '.jpg'
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(path)
    
    try:
        img = prepare_image(path)
        if img is None:
            return jsonify({'error': 'Could not read image'}), 400
            
        pred = image_model.predict(img, verbose=0)[0][0]
        
        # 0 = Fake, 1 = Real (Check your training labels!)
        # Usually standard flow_from_directory sorts alphabetical: Fake=0, Real=1
        confidence = float(pred)
        label = "REAL" if confidence > 0.5 else "FAKE"
        
        # Make confidence strictly > 50% for display
        display_conf = confidence if label == "REAL" else 1 - confidence
        
        return jsonify({'result': label, 'confidence': display_conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Give OS time to release file handle
        time.sleep(0.1)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as cleanup_err:
            # Retry after delay
            time.sleep(0.5)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

@app.route('/detect-video', methods=['POST'])
def detect_video():
    if not image_model:
        return jsonify({'error': 'Image model not active'}), 503
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    # Use unique filename
    file_ext = os.path.splitext(secure_filename(file.filename))[1] or '.mp4'
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(path)
    
    try:
        cap = cv2.VideoCapture(path)
        predictions = []
        frame_cnt = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Sample every 30th frame
            if frame_cnt % 30 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224)) # UPDATE to 224 for MobileNet
                frame = frame / 255.0
                pred = image_model.predict(np.expand_dims(frame, axis=0), verbose=0)[0][0]
                predictions.append(pred)
            frame_cnt += 1
        cap.release()
        
        if not predictions: return jsonify({'error': 'No frames extracted'}), 500
        
        avg = np.mean(predictions)
        label = "REAL" if avg > 0.5 else "FAKE"
        display_conf = float(avg) if label == "REAL" else 1 - float(avg)
        
        return jsonify({'result': label, 'confidence': display_conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Give OS time to release file handles
        time.sleep(0.2)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as cleanup_err:
            # Retry after delay
            time.sleep(0.5)
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass

@app.route('/detect-audio', methods=['POST'])
def detect_audio():
    if not audio_model:
        return jsonify({'error': 'Audio model not active'}), 503
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    unique_name = f"{uuid.uuid4()}.wav"
    path = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(path)
    
    try:
        with model_lock:
            feats = prepare_audio(path)
            pred = audio_model.predict(feats, verbose=0)[0][0]
            
        label = "REAL" if pred > 0.5 else "FAKE"
        conf = float(pred) if label == "REAL" else 1 - float(pred)
        
        return jsonify({'result': label, 'confidence': conf})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(path):
            try: os.remove(path)
            except: pass

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'image_model': 'LOADED' if image_model else 'MISSING',
        'audio_model': 'LOADED' if audio_model else 'MISSING',
        'errors': model_load_errors
    })

if __name__ == '__main__':
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', '5000')),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )