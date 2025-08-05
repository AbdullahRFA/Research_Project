import os
import tensorflow as tf
import numpy as np
import requests

# ✅ Step 1: Load pretrained MobileNet model
model = tf.keras.applications.MobileNet()
print("✅ MobileNet model loaded.")

# ✅ Step 2: Load and preprocess the image
image_path = "test_image1.jpg"  # 🔁 Replace with your image path
image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))  # Resize image
image = tf.keras.preprocessing.image.img_to_array(image)  # Convert to array
image = tf.expand_dims(image, axis=0)  # Add batch dimension
image = tf.keras.applications.mobilenet.preprocess_input(image)  # Normalize for MobileNet

# ✅ Step 3: Make prediction
prediction = model.predict(image)
label_index = tf.argmax(prediction[0]).numpy()
print(f"\n🔍 Predicted label index: {label_index}")

# ✅ Step 4: Load ImageNet class names
label_file = "imagenet_class_names.txt"

# If not already downloaded, fetch the class labels
if not os.path.exists(label_file):
    print("📥 Downloading ImageNet class names...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
    response = requests.get(url)
    with open(label_file, 'w') as f:
        f.write(response.text)

# Read class names
with open(label_file, 'r') as file:
    class_names = file.read().splitlines()

# ✅ Step 5: Map label index to class name (no need to remove 'background')
class_name = class_names[label_index]
print(f"✅ Predicted class name: {class_name}")