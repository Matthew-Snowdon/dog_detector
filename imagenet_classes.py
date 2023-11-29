import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import (ResNet50, preprocess_input,
                                                    decode_predictions)
from tensorflow.keras.preprocessing import image

# Load ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet')


# Load the image file, resizing it to 224x224 pixels (required by this model)
img_path = r'L:\dog_detector\dog.jpeg'  # Replace with your image path
img = image.load_img(img_path, target_size=(224, 224))

# Convert the image to a numpy array and preprocess it
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make a prediction
predictions = model.predict(img_array)

# Decode predictions to get class labels
decoded_predictions = decode_predictions(predictions, top=5)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")


# Open the ImageNet class index file and read the class names
with open(tf.keras.utils.get_file('imagenet_class_index.json',
                                  'https://storage.googleapis.com/download.'
                                  'tensorflow.org/data/imagenet_class_index'
                                  '.json')) as read_file:
    class_indices = json.load(read_file)
    imagenet_labels = [class_label[1] for class_label in
                       class_indices.values()]


print("Total ImageNet classes:", len(imagenet_labels))
print(imagenet_labels)
