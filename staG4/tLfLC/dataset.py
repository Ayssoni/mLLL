"""
=========================================================
Dataset Loader
FER2013 Emotion Recognition
=========================================================
"""

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.utils.class_weight import compute_class_weight

import config


# =====================================================
# TRAIN DATA AUGMENTATION
# =====================================================

train_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input,

    rotation_range=20,

    width_shift_range=0.15,

    height_shift_range=0.15,

    zoom_range=0.15,

    shear_range=0.10,

    horizontal_flip=True,

    fill_mode="nearest"
)

# =====================================================
# TEST DATA
# =====================================================

test_datagen = ImageDataGenerator(

    preprocessing_function=preprocess_input

)

# =====================================================
# LOAD TRAIN DATASET
# =====================================================

train_dataset = train_datagen.flow_from_directory(

    directory=config.TRAIN_DIR,

    target_size=(config.IMG_SIZE, config.IMG_SIZE),

    batch_size=config.BATCH_SIZE,

    class_mode="categorical",

    shuffle=True,

    seed=config.SEED
)

# =====================================================
# LOAD TEST DATASET
# =====================================================

test_dataset = test_datagen.flow_from_directory(

    directory=config.TEST_DIR,

    target_size=(config.IMG_SIZE, config.IMG_SIZE),

    batch_size=config.BATCH_SIZE,

    class_mode="categorical",

    shuffle=False
)

# =====================================================
# CLASS INFORMATION
# =====================================================

print("\n====================================")
print("Class Mapping")
print("====================================")

print(train_dataset.class_indices)

print("\n====================================")
print("Number of Classes")
print("====================================")

print(train_dataset.num_classes)

print("\n====================================")
print("Samples")
print("====================================")

print("Training :", train_dataset.samples)
print("Testing  :", test_dataset.samples)

# =====================================================
# CLASS WEIGHTS
# =====================================================

labels = train_dataset.classes

weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(labels),

    y=labels
)

class_weights = {

    i: weights[i]

    for i in range(len(weights))
}

print("\n====================================")
print("Class Weights")
print("====================================")

print(class_weights)

# =====================================================
# FUNCTION
# =====================================================

def get_datasets():

    """
    Returns

    train_dataset

    test_dataset

    class_weights
    """

    return (

        train_dataset,

        test_dataset,

        class_weights
    )