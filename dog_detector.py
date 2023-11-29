import numpy as np
import os
import shutil
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import (ResNet50,
                                                    preprocess_input,
                                                    decode_predictions)


# List of indices for dog classes in ImageNet
dog_class_indices = ['n02110627', 'n02088094', 'n02096051', 'n02107908',
                     'n02096294', 'n02110806', 'n02088238', 'n02088364',
                     'n02093647', 'n02107683', 'n02089078', 'n02086646',
                     'n02088466', 'n02088632', 'n02106166', 'n02093754',
                     'n02090622', 'n02096585', 'n02108089', 'n02112706',
                     'n02105251', 'n02101388', 'n02108422', 'n02096177',
                     'n02963159', 'n02085620', 'n02112137', 'n02101556',
                     'n02102318', 'n02106030', 'n02099429', 'n02110341',
                     'n02107142', 'n02089973', 'n02100735', 'n02102040',
                     'n02109961', 'n02099267', 'n02108915', 'n02106662',
                     'n02100236', 'n02097130', 'n02099601', 'n02101006',
                     'n02105056', 'n02091244', 'n02100877', 'n02093991',
                     'n02102973', 'n02090721', 'n02091032', 'n02085782',
                     'n02112350', 'n02105412', 'n02093859', 'n02105505',
                     'n02104029', 'n02099712', 'n02095570', 'n02111129',
                     'n02098413', 'n02110063', 'n02105162', 'n02085936',
                     'n02113978', 'n02107312', 'n02113712', 'n02097047',
                     'n02111277', 'n02094114', 'n02091467', 'n02094258',
                     'n02091635', 'n02086910', 'n02086079', 'n02113023',
                     'n02112018', 'n02110958', 'n02090379', 'n02087394',
                     'n02106550', 'n02091831', 'n02111889', 'n02104365',
                     'n02097298', 'n02092002', 'n02095889', 'n02105855',
                     'n02110185', 'n02097658', 'n02098105', 'n02093256',
                     'n02113799', 'n02097209', 'n02102480', 'n02108551',
                     'n02097474', 'n02113624', 'n02087046', 'n02100583',
                     'n02089867', 'n02092339', 'n02102177', 'n02091134',
                     'n02095314', 'n02094433']


# Load the ResNet50 model pre-trained on ImageNet data
model = ResNet50(weights='imagenet')


def classify_and_rename_image(img_path, model):
    """
    Classify an image using a pre-trained model and rename the image based
    on the top prediction label.

        Parameters: - img_path (str): The path to the image file to be
        classified and renamed. - model: A pre-trained neural network model
        capable of image classification.

        Returns: - category (str): Either 'dog' if the top prediction is a
        dog class or 'not_dog' otherwise. - new_filename (str): A new
        filename for the image, consisting of the top prediction label
        followed by the original filename.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the class
    predictions = model.predict(img_array)

    # Decode the predictions to get class labels
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    top_prediction_label = decoded_predictions[0][1]

    # Check if any of the top 5 predictions is a dog class
    is_dog = any(pred[0] in dog_class_indices for pred in decoded_predictions)
    category = 'dog' if is_dog else 'not_dog'

    # Create a new filename with the top prediction label
    new_filename = f"{top_prediction_label}_{os.path.basename(img_path)}"
    return category, new_filename


# Define paths
base_dir = r"L:\dog_detector"
new_image_directory = os.path.join(base_dir, 'unsorted')
sorted_directory_dog = os.path.join(base_dir, 'sorted', 'dog')
sorted_directory_not_dog = os.path.join(base_dir, 'sorted', 'not_dog')

# Make sure the sorted directories exist
os.makedirs(sorted_directory_dog, exist_ok=True)
os.makedirs(sorted_directory_not_dog, exist_ok=True)

# Classify and sort images
for img_name in os.listdir(new_image_directory):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(new_image_directory, img_name)
        category, new_filename = classify_and_rename_image(img_path, model)

        # Move and rename the image to the corresponding directory
        if category == 'dog':
            shutil.move(img_path, os.path.join(sorted_directory_dog,
                                               new_filename))
        else:
            shutil.move(img_path, os.path.join(sorted_directory_not_dog,
                                               new_filename))

print("Images have been sorted and renamed.")
