import cv2
import numpy as np

# Constants
IMAGE_SIZE = (224, 224)  # Resize to match model input

# Function to check if the image is a fingerprint (basic check using contours)
def is_fingerprint_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Error loading image. Please check the file path.")

    # Resize the image
    img_resized = cv2.resize(img, IMAGE_SIZE)

    # Apply thresholding to extract edges
    _, thresh = cv2.threshold(img_resized, 120, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # A basic assumption: A fingerprint should have a significant number of contours
    if len(contours) > 10:
        return True  # Likely a fingerprint
    else:
        return False  # Likely not a fingerprint

# Test the function
if __name__ == "__main__":
    image_path = input("Enter the path to the fingerprint image: ")
    if is_fingerprint_image(image_path):
        print("✅ Valid fingerprint image.")
    else:
        print("❌ Invalid fingerprint image.")
