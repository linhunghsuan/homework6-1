import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Build VGG19 Pretrained Model
base_model = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the layers of the base VGG19 model

# Build the final model on top of VGG19
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),  # Pooling to reduce spatial dimensions
    layers.Dense(1024, activation='relu'),  # Fully connected layer
    layers.Dropout(0.5),  # Dropout for regularization
    layers.Dense(2, activation='softmax')  # Output layer (2 classes)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Step 2: Load the Medical Mask Dataset (Example Path)
train_dir = r'C:\Users\user\Diva\homework6-1'  # Set the correct path for your dataset

# Image Data Generator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize the image pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Set up the training data generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Resize images to the required size (224x224 for VGG19)
    batch_size=32,
    class_mode='categorical',  # Categorical classification (2 classes)
    shuffle=True
)

# Step 3: Classify Image from URL
def test_image(image_url, model, classes):
    # Fetch the image from the URL
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Preprocess the image: resize, convert to array, and normalize
    img = img.resize((224, 224))  # Resize image to (224, 224)
    img_array = image.img_to_array(img)  # Convert image to a NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array /= 255.0  # Normalize pixel values to [0, 1]

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)

    # Map the predicted class to the class label
    predicted_class_label = classes[predicted_class[0]]

    print(f"Predicted Class: {predicted_class_label}")
    return predicted_class_label

# Example usage:
# You can replace the URL with any image URL you want to classify.
image_url ="https://na.cx/i/eqzQJYw.jpg"
test_image(image_url, model, train_generator.class_indices)
