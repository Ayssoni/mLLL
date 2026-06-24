import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIG
# =====================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_PATH = "/Users/aysoni/Documents/mLLL/dataSET1/train"
TEST_PATH = "/Users/aysoni/Documents/mLLL/dataSET1/test"

# =====================================================
# DATA AUGMENTATION
# =====================================================

train_generator = ImageDataGenerator(
    rescale=1./255,

    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,

    zoom_range=0.10,

    horizontal_flip=True
)

test_generator = ImageDataGenerator(
    rescale=1./255
)

# =====================================================
# LOAD DATASET
# =====================================================

train_data = train_generator.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_generator.flow_from_directory(
    TEST_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass Mapping:")
print(train_data.class_indices)

# =====================================================
# LOAD EFFICIENTNET
# =====================================================

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Freeze pretrained layers
base_model.trainable = False

# =====================================================
# BUILD MODEL
# =====================================================

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        3,
        activation="softmax"
    )
])

# =====================================================
# COMPILE
# =====================================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================================
# CALLBACKS
# =====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "best_efficientnet_emotion.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# =====================================================
# TRAIN
# =====================================================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS,
    callbacks=[
        early_stop,
        checkpoint
    ]
)

# =====================================================
# SAVE MODEL
# =====================================================

model.save("emotion_stage4.h5")

print("\nModel Saved Successfully")

# =====================================================
# EVALUATE
# =====================================================

loss, accuracy = model.evaluate(test_data)

print(f"\nTest Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

# =====================================================
# ACCURACY GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend([
    "Train Accuracy",
    "Validation Accuracy"
])

plt.grid(True)
plt.show()

# =====================================================
# LOSS GRAPH
# =====================================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend([
    "Train Loss",
    "Validation Loss"
])

plt.grid(True)
plt.show()

# =====================================================
# PREDICTIONS
# =====================================================

predictions = model.predict(test_data)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_data.classes

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=list(test_data.class_indices.keys()),
    yticklabels=list(test_data.class_indices.keys())
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=list(test_data.class_indices.keys())
    )
)