import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load trained model
model_path = 'C:/Anusha/blood_group_prediction/models/blood_type_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

model = load_model(model_path)
print("✅ Model loaded successfully.")

# Define blood groups based on class indices
blood_groups = ['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-']

# Function to preprocess input image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image file not found at {image_path}")

    img = cv2.imread(image_path)  # Read image
    img = cv2.resize(img, (128, 128))  # Resize to match model input shape
    img = img_to_array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict blood type
def predict_blood_type(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    return blood_groups[predicted_class], confidence

# Test the prediction (Change path to actual fingerprint image)
image_path = 'C:/Anusha/blood_group_prediction/models/test_fingerprint.jpg'
try:
    predicted_blood_group, confidence = predict_blood_type(image_path)
    print(f"✅ Predicted Blood Group: {predicted_blood_group} (Confidence: {confidence:.2f}%)")
except FileNotFoundError as e:
    print(e)
