import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Constants
DATASET_PATH = 'C:/blood_group_prediction/dataset'  # Path to your dataset
IMAGE_SIZE = (224, 224)  # Resize images to 224x224 for DenseNet
CLASSES = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']  # Your class names
BATCH_SIZE = 32
EPOCHS = 20

# Map class names to numeric labels
label_map = {class_name: index for index, class_name in enumerate(CLASSES)}

# Load image paths and labels
image_paths, labels = [], []
for class_name in CLASSES:
    class_folder = os.path.join(DATASET_PATH, class_name)
    print(f"Checking folder: {class_folder}")
    
    # Find all image files (jpg, png, jpeg, bmp)
    image_files = glob(os.path.join(class_folder, '*.jpg')) + \
                  glob(os.path.join(class_folder, '*.png')) + \
                  glob(os.path.join(class_folder, '*.jpeg')) + \
                  glob(os.path.join(class_folder, '*.bmp'))
    
    if len(image_files) == 0:
        print(f"Warning: No images found in {class_folder}")
    
    for img_path in image_files:
        image_paths.append(img_path)
        labels.append(label_map[class_name])

# Check if images are loaded
print(f"Total images found: {len(image_paths)}")

# If no images found, raise an error
if len(image_paths) == 0:
    raise ValueError("No images found in the dataset. Please check your dataset path and image files.")

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Prepare training and validation data
X_train_data = np.vstack([load_and_preprocess_image(img_path) for img_path in X_train])
X_val_data = np.vstack([load_and_preprocess_image(img_path) for img_path in X_val])

# Convert labels to one-hot encoding
y_train_data = np.array(y_train)
y_val_data = np.array(y_val)

# Load DenseNet121 model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(CLASSES), activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of DenseNet121 initially
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(X_train_data, y_train_data, epochs=EPOCHS, validation_data=(X_val_data, y_val_data), batch_size=BATCH_SIZE, callbacks=[early_stopping])

# Unfreeze some layers for fine-tuning
for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
    layer.trainable = True

# Recompile the model after unfreezing layers
model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Continue training after fine-tuning
model.fit(X_train_data, y_train_data, epochs=10, validation_data=(X_val_data, y_val_data), batch_size=BATCH_SIZE)

# Save the model in .keras format (recommended)
model.save('blood_group_predictor_model.keras')

