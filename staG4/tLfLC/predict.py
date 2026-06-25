"""
=========================================================
Single Image Emotion Prediction
=========================================================
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

import config

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading Model...")

model = tf.keras.models.load_model(
    config.FINAL_MODEL
)

print("Model Loaded Successfully!")

# =========================================================
# IMAGE PATH
# =========================================================

IMAGE_PATH = "/Users/aysoni/Documents/mLLL/test.jpg"

# Change this path to your own image

# =========================================================
# CHECK IMAGE
# =========================================================

if not os.path.exists(IMAGE_PATH):

    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )

# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    raise ValueError(
        "OpenCV could not load the image."
    )

# OpenCV loads images in BGR
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

original = image.copy()

# =========================================================
# PREPROCESS
# =========================================================

image = cv2.resize(

    image,

    (
        config.IMG_SIZE,
        config.IMG_SIZE
    )

)

image = image.astype("float32")

image = preprocess_input(image)

image = np.expand_dims(

    image,

    axis=0

)

# =========================================================
# PREDICTION
# =========================================================

prediction = model.predict(image, verbose=0)

prediction = prediction[0]

predicted_index = np.argmax(prediction)

emotion = config.CLASS_NAMES[predicted_index]

confidence = prediction[predicted_index] * 100

# =========================================================
# PRINT RESULT
# =========================================================

print("\n============================")

print(f"Prediction : {emotion}")

print(f"Confidence : {confidence:.2f}%")

print("============================\n")

# =========================================================
# ALL PROBABILITIES
# =========================================================

print("Class Probabilities\n")

for label, prob in zip(

    config.CLASS_NAMES,

    prediction

):

    print(f"{label:10s} : {prob*100:.2f}%")

# =========================================================
# DISPLAY IMAGE
# =========================================================

plt.figure(figsize=(6,6))

plt.imshow(original)

plt.axis("off")

plt.title(

    f"{emotion} ({confidence:.2f}%)"

)

plt.show()

# =========================================================
# BAR GRAPH
# =========================================================

plt.figure(figsize=(10,5))

plt.bar(

    config.CLASS_NAMES,

    prediction*100

)

plt.title("Prediction Confidence")

plt.ylabel("Confidence (%)")

plt.ylim([0,100])

plt.grid(axis="y")

plt.show()