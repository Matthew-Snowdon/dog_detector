import os
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define paths
base_dir = r"L:\dog_detector"
data_dir = os.path.join(base_dir, 'dataset')  # base directory

# Load ResNet50 as the base model, with pre-trained ImageNet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,
                                                                          224,
                                                                          3))
base_model.trainable = False  # Freeze the base model layers

# Add new top layers for binary classification
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # 1 unit for binary classification
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=os.path.join(base_dir, 'dog_classifier_resnet_model_aug.keras'),
    monitor='val_loss',
    save_best_only=True
)

# Data generators for training and validation with automatic splitting
datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),  # Adjusted for ResNet50
    batch_size=32,
    class_mode='binary',  # Binary classification
    subset='training'  # Set as training data
)

# Create an instance of ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Flow validation images in batches using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples
                     // validation_generator.batch_size,
    epochs=10,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the trained model
model.save(os.path.join(base_dir, 'dog_classifier_resnet_model_aug.keras'))
