import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model('victim_detection_model.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed format
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return jsonify({'error': 'Unsupported file format'})

    # Preprocess the image
    img = image.load_img(file, target_size=(150, 150))
    img_array = preprocess_image(img)

    # Make prediction
    prediction = model.predict(img_array)
    prediction_label = "Victim Detected" if prediction[0] > 0.5 else "No Victim Detected"

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
