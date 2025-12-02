import os
import json
import tempfile
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = 'your-secret-key-here'  # Change this in production

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Те же настройки, что и в train-скрипте
IMG_SIZE = 224
OUTPUT_DIR = Path("ml/outputs_dogs")

# Определяем устройство
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Using device: {DEVICE}")

# Трансформ для инференса (как val/test в обучении)
test_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    for p in model.parameters():
        p.requires_grad = False

    in_feats = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feats, num_classes),
    )
    return model


def load_for_inference(
    model_path: Path = OUTPUT_DIR / "best_model.pt",
    classes_path: Path = OUTPUT_DIR / "classes.json",
):
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл модели: {model_path}")
    if not classes_path.exists():
        raise FileNotFoundError(f"Не найден файл классов: {classes_path}")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names: List[str] = json.load(f)

    num_classes = len(class_names)
    model = build_model(num_classes)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, class_names


def prettify_breed(class_name: str) -> str:
    if "-" in class_name:
        class_name = class_name.split("-", 1)[1]
    class_name = class_name.replace("_", " ")
    return class_name


@torch.no_grad()
def predict_image(model, class_names, img_path, topk=5):
    img = Image.open(img_path).convert("RGB")
    x = test_tfms(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    vals, idxs = probs.topk(topk)

    results = []
    for v, i in zip(vals.cpu().tolist(), idxs.cpu().tolist()):
        breed_name = prettify_breed(class_names[i])  # <-- вот тут
        results.append((breed_name, v))
    return results


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Load model and class names at startup
try:
    model, class_names = load_for_inference()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, class_names = None, []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # If user does not select file, browser submits empty part without filename
        if file.filename == '':
            flash('No selected file')
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            try:
                # Save the file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(temp_path)
                
                # Make prediction
                if model is None:
                    return jsonify({'error': 'Model not loaded properly'}), 500
                
                predictions = predict_image(model, class_names, temp_path, topk=5)
                
                # Convert probabilities to percentages
                results = []
                for breed, prob in predictions:
                    percentage = round(prob * 100, 2)
                    results.append({
                        'breed': breed,
                        'probability': percentage
                    })
                
                # Clean up temporary file
                os.remove(temp_path)
                
                return jsonify({
                    'success': True,
                    'predictions': results
                })
                
            except Exception as e:
                # Clean up temporary file if exists
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG files are allowed.'}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)