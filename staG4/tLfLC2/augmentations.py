"""
=========================================================
Emotion Recognition V2
Data Augmentation
=========================================================
"""

import tensorflow as tf
import config


# ==========================================================
# TRAINING AUGMENTATION
# ==========================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(

        mode="horizontal"

    ),

    tf.keras.layers.RandomRotation(

        factor=config.ROTATION_FACTOR

    ),

    tf.keras.layers.RandomZoom(

        height_factor=config.ZOOM_FACTOR,

        width_factor=config.ZOOM_FACTOR

    ),

    tf.keras.layers.RandomTranslation(

        height_factor=config.TRANSLATION_FACTOR,

        width_factor=config.TRANSLATION_FACTOR

    ),

    tf.keras.layers.RandomContrast(

        factor=config.CONTRAST_FACTOR

    )

], name="DataAugmentation")


# ==========================================================
# APPLY AUGMENTATION
# ==========================================================

def augment(images, labels):

    images = data_augmentation(

        images,

        training=True

    )

    return images, labels


# ==========================================================
# APPLY TO TRAIN DATASET
# ==========================================================

def apply_augmentation(train_dataset):

    train_dataset = train_dataset.map(

        augment,

        num_parallel_calls=tf.data.AUTOTUNE

    )

    return train_dataset


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(data_augmentation.summary())