"""
=========================================================
Emotion Recognition V2
Dataset Pipeline (tf.data)
=========================================================
"""

import tensorflow as tf
import config

AUTOTUNE = tf.data.AUTOTUNE


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(dataset_path, shuffle=True):

    dataset = tf.keras.utils.image_dataset_from_directory(

        dataset_path,

        labels="inferred",

        label_mode="categorical",

        class_names=config.CLASS_NAMES,

        image_size=(config.IMAGE_SIZE, config.IMAGE_SIZE),

        batch_size=config.BATCH_SIZE,

        shuffle=shuffle,

        seed=config.SEED

    )

    return dataset


# ==========================================================
# OPTIMIZE DATASET
# ==========================================================

def optimize_dataset(dataset, training=False):

    if training:

        dataset = dataset.shuffle(

            buffer_size=config.SHUFFLE_BUFFER,

            seed=config.SEED

        )

    dataset = dataset.cache()

    dataset = dataset.prefetch(

        AUTOTUNE

    )

    return dataset


# ==========================================================
# LOAD ALL DATASETS
# ==========================================================

def get_datasets():

    print("\nLoading Dataset...\n")

    train_dataset = load_dataset(

        config.FINAL_DATASET / config.TRAIN_FOLDER,

        shuffle=True

    )

    valid_dataset = load_dataset(

        config.FINAL_DATASET / config.VALID_FOLDER,

        shuffle=False

    )

    test_dataset = load_dataset(

        config.FINAL_DATASET / config.TEST_FOLDER,

        shuffle=False

    )

    train_dataset = optimize_dataset(

        train_dataset,

        training=True

    )

    valid_dataset = optimize_dataset(

        valid_dataset,

        training=False

    )

    test_dataset = optimize_dataset(

        test_dataset,

        training=False

    )

    print("\nDataset Loaded Successfully\n")

    print("Class Names")

    print(train_dataset.class_names)

    return (

        train_dataset,

        valid_dataset,

        test_dataset

    )


# ==========================================================
# INFORMATION
# ==========================================================

def dataset_info(dataset):

    print("\nDataset Information")

    print("-" * 40)

    for images, labels in dataset.take(1):

        print("Images Shape :", images.shape)

        print("Labels Shape :", labels.shape)

        print("Image dtype  :", images.dtype)

        print("Label dtype  :", labels.dtype)

        print("Min Pixel    :", tf.reduce_min(images).numpy())

        print("Max Pixel    :", tf.reduce_max(images).numpy())


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    train_ds, valid_ds, test_ds = get_datasets()

    dataset_info(train_ds)