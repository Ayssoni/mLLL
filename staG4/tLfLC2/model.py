"""
=========================================================
Emotion Recognition V2
Model Definition
=========================================================
"""

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import Model

import config
from augmentations import data_augmentation


# ==========================================================
# BACKBONE
# ==========================================================

def get_backbone():

    backbone = tf.keras.applications.EfficientNetV2B2(

        include_top=False,

        weights="imagenet",

        input_shape=config.IMAGE_SHAPE

    )

    backbone.trainable = False

    return backbone


# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model():

    backbone = get_backbone()

    inputs = layers.Input(

        shape=config.IMAGE_SHAPE,

        name="Input_Image"

    )

    # ------------------------------------------------------
    # Data Augmentation
    # ------------------------------------------------------

    x = data_augmentation(inputs)

    # ------------------------------------------------------
    # Rescaling
    # ------------------------------------------------------

    x = layers.Rescaling(

        scale=1.0 / 255.0,

        name="Rescaling"

    )(x)

    # ------------------------------------------------------
    # Backbone
    # ------------------------------------------------------

    x = backbone(

        x,

        training=False

    )

    # ------------------------------------------------------
    # Pooling
    # ------------------------------------------------------

    x = layers.GlobalAveragePooling2D(

        name="GlobalAveragePooling"

    )(x)

    # ------------------------------------------------------
    # Dense Block 1
    # ------------------------------------------------------

    x = layers.BatchNormalization()(x)

    x = layers.Dense(

        config.DENSE_1,

        activation="relu"

    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(

        config.DROPOUT_1

    )(x)

    # ------------------------------------------------------
    # Dense Block 2
    # ------------------------------------------------------

    x = layers.Dense(

        config.DENSE_2,

        activation="relu"

    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(

        config.DROPOUT_2

    )(x)

    # ------------------------------------------------------
    # Output
    # ------------------------------------------------------

    outputs = layers.Dense(

        config.NUM_CLASSES,

        activation="softmax",

        name="Emotion"

    )(x)

    model = Model(

        inputs,

        outputs,

        name="EmotionRecognitionV2"

    )

    return model, backbone


# ==========================================================
# UNFREEZE BACKBONE
# ==========================================================

def unfreeze_backbone(

        backbone,

        last_layers

):

    backbone.trainable = True

    for layer in backbone.layers[:-last_layers]:

        layer.trainable = False

    print(

        f"\nUnfroze last {last_layers} layers."

    )


# ==========================================================
# MODEL SUMMARY
# ==========================================================

if __name__ == "__main__":

    model, backbone = build_model()

    model.summary()

    print("\n")

    print(

        "Total Layers :", len(backbone.layers)

    )