import os
import json
import numpy as np
import cv2
import uuid
import base64
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max limit

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class names
MODEL_PATH = 'model/mobilenet_mask_model.keras'
CLASS_NAMES_PATH = 'model/class_names.json'

model = None
class_names = None

def load_resources():
    global model, class_names
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Error: Model file not found at {MODEL_PATH}")
            
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, 'r') as f:
                class_names = json.load(f)
            print(f"Loaded classes: {class_names}")
        else:
            print(f"Error: Class names file not found at {CLASS_NAMES_PATH}")
    except Exception as e:
        print(f"Error loading model or class names: {e}")

load_resources()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def process_prediction(image):
    """Common prediction logic matching realtime_detection.py exactly"""
    # 2. Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. Resize image to (224, 224)
    image_resized = cv2.resize(image_rgb, (224, 224))
    
    # 4. Convert to numpy array (ensure float32)
    image_array = np.array(image_resized, dtype=np.float32)
    
    # 5. Expand dimension
    image_expanded = np.expand_dims(image_array, axis=0)
    
    # IMPORTANT FIX: In train_mobilenet.py, tf.keras.applications.mobilenet_v2.preprocess_input
    # is built directly into the model graph as a layer (x = preprocess_input(inputs)).
    # Therefore, we MUST NOT call it manually here, otherwise the image gets preprocessed TWICE,
    # which leads to incorrect predictions. We just pass the 0-255 RGB array like realtime_detection.py!
    
    # 7. Predict using model.predict
    predictions = model.predict(image_expanded, verbose=0)
    
    # 8. Get class_id using np.argmax
    predicted_class_idx = int(np.argmax(predictions))
    
    # 9. Get confidence using np.max * 100
    confidence = float(np.max(predictions)) * 100
    
    # 10. Get label using class_names
    if isinstance(class_names, dict):
        if str(predicted_class_idx) in class_names:
            predicted_class_name = class_names[str(predicted_class_idx)]
        else:
             predicted_class_name = list(class_names.values())[predicted_class_idx]
    else:
        predicted_class_name = class_names[predicted_class_idx]

    # Debug output in terminal
    print("-" * 30)
    print(f"Prediction probabilities: {predictions}")
    print(f"Predicted class id: {predicted_class_idx}")
    print(f"Predicted label: {predicted_class_name}")
    print(f"Confidence: {confidence:.2f}%")
    print("-" * 30)

    return predicted_class_name, confidence

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or class_names is None:
        return jsonify({'error': 'Server Error: Model or class names not loaded properly.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded. Please upload an image.'}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG image.'}), 400

    filepath = ""
    try:
        # Save file to uploads folder
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # 1. Read uploaded image using cv2.imread(filepath)
        image = cv2.imread(filepath)
        
        if image is None:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': 'Could not read the uploaded image file.'}), 400
            
        predicted_class_name, confidence = process_prediction(image)

        return jsonify({
            'success': True,
            'label': predicted_class_name,
            'confidence': f"{confidence:.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
    finally:
        # Clean up the file after prediction to save disk space
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                pass

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    if model is None or class_names is None:
        return jsonify({'error': 'Server Error: Model or class names not loaded properly.'}), 500

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided.'}), 400
        
    try:
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        
        # 1. Read uploaded image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Could not decode image.'}), 400
            
        predicted_class_name, confidence = process_prediction(image)

        return jsonify({
            'success': True,
            'label': predicted_class_name,
            'confidence': f"{confidence:.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
