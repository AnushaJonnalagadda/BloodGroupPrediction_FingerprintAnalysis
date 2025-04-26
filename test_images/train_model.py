import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    'C:/Anusha/blood_group_prediction/dataset/train',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = datagen.flow_from_directory(
    'C:/Anusha/blood_group_prediction/dataset/val',
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Load ResNet50 base model without top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model layers

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global pooling instead of Flatten
x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(8, activation='softmax')(x)  # Ensure `outputs` is a Keras tensor

# Define the model correctly
model = Model(inputs=base_model.input, outputs=outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the model
model.save('C:/Anusha/blood_group_prediction/models/blood_type_model.h5')

print("✅ Model training complete and saved successfully!")
