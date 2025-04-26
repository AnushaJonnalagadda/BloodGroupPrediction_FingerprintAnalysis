import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Constants
IMAGE_SIZE = (224, 224)  # Input size for DenseNet
MODEL_PATH = 'blood_group_predictor_model.keras'  # Updated to Keras format
LABELS = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Load the trained model
model = load_model(MODEL_PATH)

# Function to preprocess fingerprint image
def preprocess_fingerprint_image(img_path):
    # Load image in grayscale for blood group prediction
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")

    # Resize the image to match the model input size
    img = cv2.resize(img, IMAGE_SIZE)

    # Convert to 3 channels to match model input (as DenseNet expects 3 channels)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Convert image to array and normalize it
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values

    return img_array

# Function to predict blood group
def predict_blood_group(img_path):
    if not os.path.isfile(img_path):
        print("File does not exist. Please check the path.")
        return

    img_array = preprocess_fingerprint_image(img_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_blood_group = LABELS[predicted_class_index]

    print(f" The predicted blood group is: {predicted_blood_group}")

# Main
if __name__ == "__main__":
    img_path = input(" Enter the path to the fingerprint image: ")
    predict_blood_group(img_path)
