import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def get_data_generators(data_dir, img_size=(224,224), batch_size=32, validation_split=0.2, seed=42):
    """
    Assumes your data directory has subfolders per class:
      data_dir/
        ├── class_A/
        └── class_B/
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split,
        horizontal_flip=True,
        rotation_range=15
    )
    test_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed
    )
    val_gen = test_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=seed
    )
    return train_gen, val_gen

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)

def load_model(path):
    from tensorflow.keras.models import load_model
    return load_model(path)
