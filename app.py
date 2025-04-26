import os
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model (ensure the path is correct)
MODEL_PATH = 'C:\\blood_group_prediction\\model\\blood_group_predictor_model.keras'
model = load_model(MODEL_PATH)

# Constants
IMAGE_SIZE = (224, 224)  # Input size for DenseNet
LABELS = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Function to preprocess the image
def preprocess_image(img_path):
    # Load the image and resize it
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")
    
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Convert to 3 channels to match model input
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Convert the image to array and normalize it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the pixel values
    
    return img_array

# Function to predict blood group
def predict_blood_group(img_path):
    # Preprocess the image before prediction
    img_array = preprocess_image(img_path)
    
    # Predict the class
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_blood_group = LABELS[predicted_class_index]
    
    return predicted_blood_group

# Route for the homepage (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        # Save the file temporarily
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        try:
            # Predict the blood group
            predicted_blood_group = predict_blood_group(file_path)
            return jsonify({"predicted_blood_group": predicted_blood_group})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Route for the logout (for demo purposes)
@app.route('/logout')
def logout():
    return render_template('login.html')  # Redirect to login page (adjust URL as needed)

# Run the Flask app
if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    app.run(debug=True)
