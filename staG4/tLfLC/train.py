"""
=========================================================
Emotion Recognition Training
Feature Extraction
Fine Tuning
=========================================================
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import (

    EarlyStopping,

    ModelCheckpoint,

    ReduceLROnPlateau,

    TensorBoard,

    CSVLogger

)

import config

from dataset import get_datasets

from model import (

    build_model,

    compile_model,

    model_summary

)

# =========================================================
# RANDOM SEED
# =========================================================

random.seed(config.SEED)

np.random.seed(config.SEED)

tf.random.set_seed(config.SEED)

# =========================================================
# GPU MEMORY GROWTH
# =========================================================

gpus = tf.config.experimental.list_physical_devices("GPU")

if len(gpus) > 0:

    try:

        for gpu in gpus:

            tf.config.experimental.set_memory_growth(

                gpu,

                True

            )

        print("\nGPU Ready\n")

    except RuntimeError as e:

        print(e)

else:

    print("\nRunning on CPU\n")

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading Dataset...\n")

train_dataset, test_dataset, class_weights = get_datasets()

print("\nDataset Loaded Successfully\n")

# =========================================================
# BUILD MODEL
# =========================================================

print("\nBuilding Model...\n")

model, base_model = build_model()

compile_model(

    model,

    config.INITIAL_LR

)

model_summary(model)

# =========================================================
# CALLBACKS
# =========================================================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=config.EARLY_STOPPING_PATIENCE,

    restore_best_weights=True,

    verbose=1

)

checkpoint = ModelCheckpoint(

    filepath=config.BEST_MODEL,

    monitor="val_accuracy",

    save_best_only=True,

    save_weights_only=False,

    mode="max",

    verbose=1

)

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=config.REDUCE_FACTOR,

    patience=config.REDUCE_LR_PATIENCE,

    min_lr=config.MIN_LR,

    verbose=1

)

tensorboard = TensorBoard(

    log_dir=config.TENSORBOARD_LOGS

)

csv_logger = CSVLogger(

    os.path.join(

        config.RESULT_DIR,

        "training_log.csv"

    )

)

callbacks = [

    early_stop,

    checkpoint,

    reduce_lr,

    tensorboard,

    csv_logger

]

print("\nEverything Ready\n")

# =========================================================
# FEATURE EXTRACTION TRAINING
# =========================================================

print("\n==========================================")
print("Starting Feature Extraction Training")
print("==========================================\n")

history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=config.INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)

print("\n==========================================")
print("Feature Extraction Completed")
print("==========================================\n")

# =========================================================
# SAVE FEATURE EXTRACTION MODEL
# =========================================================

feature_model_path = os.path.join(

    config.MODEL_DIR,

    "feature_extraction_model.keras"

)

model.save(feature_model_path)

print(f"\nFeature Extraction Model Saved : {feature_model_path}")

# =========================================================
# SAVE TRAINING HISTORY
# =========================================================

history_path = os.path.join(

    config.RESULT_DIR,

    "history.npy"

)

np.save(

    history_path,

    history.history

)

print("\nTraining History Saved")

# =========================================================
# ACCURACY GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(

    history.history["accuracy"],

    linewidth=2,

    label="Training Accuracy"

)

plt.plot(

    history.history["val_accuracy"],

    linewidth=2,

    label="Validation Accuracy"

)

plt.title("Feature Extraction Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

plt.savefig(

    os.path.join(

        config.RESULT_DIR,

        "accuracy.png"

    )

)

plt.show()

# =========================================================
# LOSS GRAPH
# =========================================================

plt.figure(figsize=(10,6))

plt.plot(

    history.history["loss"],

    linewidth=2,

    label="Training Loss"

)

plt.plot(

    history.history["val_loss"],

    linewidth=2,

    label="Validation Loss"

)

plt.title("Feature Extraction Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.savefig(

    os.path.join(

        config.RESULT_DIR,

        "loss.png"

    )

)

plt.show()

print("\nGraphs Saved Successfully")
# ==========================================================
# FINE TUNING
# ==========================================================

print("\n========================================")
print("Starting Fine-Tuning")
print("========================================\n")

# Unfreeze EfficientNet
base_model.trainable = True

# Freeze lower layers
for layer in base_model.layers[:-config.UNFREEZE_LAST]:
    layer.trainable = False

print(f"Fine-tuning last {config.UNFREEZE_LAST} layers")

# ==========================================================
# COMPILE AGAIN
# ==========================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(

        learning_rate=config.FINE_TUNE_LR

    ),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print("\nModel Recompiled\n")

# ==========================================================
# CONTINUE TRAINING
# ==========================================================

fine_history = model.fit(

    train_dataset,

    validation_data=test_dataset,

    epochs=config.TOTAL_EPOCHS,

    initial_epoch=config.INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=callbacks,

    verbose=1

)

print("\nFine-Tuning Completed\n")

# ==========================================================
# SAVE FINAL MODEL
# ==========================================================

model.save(config.FINAL_MODEL)

print("\nFinal Model Saved")

print(config.FINAL_MODEL)

# ==========================================================
# COMBINE HISTORY
# ==========================================================

acc = history.history["accuracy"] + \
      fine_history.history["accuracy"]

val_acc = history.history["val_accuracy"] + \
          fine_history.history["val_accuracy"]

loss = history.history["loss"] + \
       fine_history.history["loss"]

val_loss = history.history["val_loss"] + \
           fine_history.history["val_loss"]

plt.figure(figsize=(10,6))

plt.plot(acc, linewidth=2)

plt.plot(val_acc, linewidth=2)

plt.axvline(

    x=config.INITIAL_EPOCHS-1,

    color="red",

    linestyle="--",

    label="Fine Tune Starts"

)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend([

    "Train",

    "Validation",

    "Fine Tune"

])

plt.savefig(

    os.path.join(

        config.RESULT_DIR,

        "final_accuracy.png"

    )

)

plt.show()

plt.figure(figsize=(10,6))

plt.plot(loss, linewidth=2)

plt.plot(val_loss, linewidth=2)

plt.axvline(

    x=config.INITIAL_EPOCHS-1,

    color="red",

    linestyle="--",

    label="Fine Tune Starts"

)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.legend([

    "Train",

    "Validation",

    "Fine Tune"

])

plt.savefig(

    os.path.join(

        config.RESULT_DIR,

        "final_loss.png"

    )

)

plt.show()

print("\nTraining Completed Successfully")