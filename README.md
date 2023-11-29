# Image Classification and Sorting with ResNet50

This Python script uses a pre-trained [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) model to classify images and sort them into "dog" and "not_dog" categories based on their top prediction label. 
It is useful for sorting a collection of images, determining if they contain dogs or not, and renaming them accordingly.

## Requirements

To run this script, you'll need:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- shutil (Python standard library)
- [ResNet50 pre-trained weights](https://keras.io/api/applications/resnet/#resnet50-function) (automatically downloaded when using `ResNet50(weights='imagenet')`)

## Usage

1. Clone or download this repository to your local machine.

2. Install the required Python packages if you haven't already:

   ```bash
   pip install tensorflow keras numpy

 Place the images you want to classify and sort into the unsorted directory within the project directory.

Run the script:

python classify_and_sort_images.py

The script will classify each image and move it to either the sorted/dog or sorted/not_dog directory based on its classification result.

After running the script, you'll find your sorted images in the sorted directory.

Customization
You can customize the script by modifying the dog_class_indices list to include specific ImageNet class indices related to dogs. The script checks if any of the top 5 predictions match these indices to determine if an image is classified as a "dog."

Acknowledgments
The ResNet50 model used in this script is pre-trained on the ImageNet dataset.
ImageNet class indices for dogs are included in the dog_class_indices list.
Keras and TensorFlow are used for deep learning tasks.
