"""
=========================================================
Model Architecture
FER2013 Emotion Recognition

Backbone:
    EfficientNetB0 (ImageNet)

Author:
    Ayush

=========================================================
"""

import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)

from tensorflow.keras.models import Model

import config


# =========================================================
# BUILD MODEL
# =========================================================

def build_model():

    inputs = Input(
        shape=(
            config.IMG_SIZE,
            config.IMG_SIZE,
            3
        ),
        name="Input_Image"
    )

    # -----------------------------------------------------
    # Backbone
    # -----------------------------------------------------

    base_model = EfficientNetB0(

        include_top=False,

        weights="imagenet",

        input_tensor=inputs
    )

    # Freeze entire EfficientNet
    base_model.trainable = False

    # -----------------------------------------------------
    # Classification Head
    # -----------------------------------------------------

    x = base_model.output

    x = GlobalAveragePooling2D(
        name="GlobalAveragePooling"
    )(x)

    x = BatchNormalization(
        name="BatchNorm_1"
    )(x)

    x = Dense(
        config.DENSE_1,
        activation="relu",
        name="Dense_512"
    )(x)

    x = Dropout(
        config.DROPOUT_1,
        name="Dropout_1"
    )(x)

    x = Dense(
        config.DENSE_2,
        activation="relu",
        name="Dense_256"
    )(x)

    x = Dropout(
        config.DROPOUT_2,
        name="Dropout_2"
    )(x)

    outputs = Dense(

        config.NUM_CLASSES,

        activation="softmax",

        name="Output"

    )(x)

    model = Model(

        inputs=inputs,

        outputs=outputs,

        name="EmotionRecognition"

    )

    return model, base_model


# =========================================================
# COMPILE MODEL
# =========================================================

def compile_model(model, learning_rate):

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(

        optimizer=optimizer,

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]

    )

    return model


# =========================================================
# UNFREEZE MODEL
# =========================================================

def unfreeze_model(base_model):

    print("\n===================================")
    print("Fine Tuning Enabled")
    print("===================================\n")

    base_model.trainable = True

    # Freeze lower layers

    for layer in base_model.layers[:-config.UNFREEZE_LAST]:

        layer.trainable = False

    # Keep upper layers trainable

    for layer in base_model.layers[-config.UNFREEZE_LAST:]:

        layer.trainable = True

    print(

        f"Trainable Layers : {config.UNFREEZE_LAST}"

    )

    return base_model


# =========================================================
# PRINT MODEL INFO
# =========================================================

def model_summary(model):

    print("\n===================================")
    print("MODEL SUMMARY")
    print("===================================\n")

    model.summary()

    print("\nTotal Layers :", len(model.layers))

    print("\nTrainable Parameters :")

    trainable = 0

    for variable in model.trainable_variables:

        trainable += tf.size(variable).numpy()

    print(trainable)

    print("\n===================================\n")