from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import json

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- LOAD MODEL & CLASS MAP ---
# Load model once at startup (Efficient!)
print("Loading Model...")
model = None

def get_model():
    global model
    if model is None:
        # Only load the model when we actually need it
        model = load_model('animal_breed_model.h5')
        print("Model loaded successfully!")
    return model
#model = load_model('animal_breed_model.h5')

with open('class_indices.json', 'r') as f:
    indices = json.load(f)
# Invert: {0: 'beagle'}
labels = {v: k for k, v in indices.items()}

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    # Show the homepage
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        # 1. Save the file temporarily
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # 2. Process Image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Add this line inside your function before prediction
        model = get_model()

        # Then your existing code runs...
        pred = model.predict(img_array)
        # 3. Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        predicted_breed = labels[predicted_index].replace('_', ' ').title()

        # 4. Return result to the Frontend
        return render_template('index.html', 
                               result=predicted_breed, 
                               confidence=f"{confidence:.2f}", 
                               image_url=filepath)

if __name__ == '__main__':

    app.run(debug=True)
