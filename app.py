import os
import json
import numpy as np
import gc # Garbage Collection for memory safety
from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- 1. CONFIGURATION ---
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- 2. LOAD CLASS MAPPINGS ---
# We load this ONCE at the top.
# This ensures your model knows "0" = "Beagle", "1" = "Bulldog", etc.
if os.path.exists('class_indices.json'):
    with open('class_indices.json', 'r') as f:
        indices = json.load(f)
    # Invert dictionary to get {0: 'beagle', 1: 'bulldog'}
    labels = {v: k for k, v in indices.items()}
else:
    print("WARNING: class_indices.json not found! Predictions might fail.")
    labels = {}

# --- 3. LAZY LOADING MODEL (Memory Fix) ---
model = None

def get_model():
    global model
    if model is None:
        print("⏳ Loading Model into Memory...")
        model = load_model('animal_breed_model.h5')
        print("✅ Model loaded successfully!")
    return model

# --- 4. ROUTES ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    uploaded_image_url = None
    error_message = None

    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return render_template('index.html', error="No file part")
            
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error="No selected file")

            if file:
                # A. Save the file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # B. Prepare Image (MobileNet Style)
                img = image.load_img(filepath, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array) # Restored your original preprocessing!

                # C. Load Model & Predict
                loaded_model = get_model()
                predictions = loaded_model.predict(img_array)
                
                # D. Decode Result
                predicted_index = np.argmax(predictions)
                confidence = np.max(predictions) * 100
                
                # Get Breed Name from JSON labels
                breed_name = labels.get(predicted_index, "Unknown Breed").replace('_', ' ').title()
                
                # Format the result string
                prediction_text = f"{breed_name} ({confidence:.2f}%)"
                
                # Create URL for the image to display it
                uploaded_image_url = url_for('static', filename=f'uploads/{filename}')

                # E. MEMORY CLEANUP (Critical for Free Tier)
                del img_array
                del predictions
                gc.collect()

        except Exception as e:
            print(f"Error: {e}")
            error_message = f"An error occurred: {str(e)}"

    # Render template with all variables
    return render_template('index.html', 
                           prediction=prediction_text, 
                           image_url=uploaded_image_url,
                           error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
