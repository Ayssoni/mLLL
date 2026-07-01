"""
=========================================================
Emotion Recognition V2
Single Image Prediction
=========================================================
"""

from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import config


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Model...\n")

model = tf.keras.models.load_model(

    config.FINAL_MODEL,

    compile=False

)

print("Model Loaded Successfully.\n")


# ==========================================================
# IMAGE PATH
# ==========================================================

IMAGE_PATH = r"/Users/aysoni/Desktop/test.jpg"

# Change this to your image


# ==========================================================
# CHECK IMAGE
# ==========================================================

if not Path(IMAGE_PATH).exists():

    raise FileNotFoundError(

        f"\nImage Not Found:\n{IMAGE_PATH}"

    )


# ==========================================================
# READ IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    raise RuntimeError("Unable to read image.")


original = image.copy()

image = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2RGB

)


# ==========================================================
# PREPROCESS
# ==========================================================

image = cv2.resize(

    image,

    (

        config.IMAGE_SIZE,

        config.IMAGE_SIZE

    )

)

image = image.astype(np.float32)

image = np.expand_dims(

    image,

    axis=0

)


# ==========================================================
# PREDICT
# ==========================================================

prediction = model.predict(

    image,

    verbose=0

)

emotion_index = np.argmax(

    prediction

)

confidence = prediction[0][emotion_index]

emotion = config.CLASS_NAMES[emotion_index]


# ==========================================================
# PRINT RESULT
# ==========================================================

print("=" * 50)

print(f"Prediction : {emotion}")

print(f"Confidence : {confidence:.2%}")

print("=" * 50)


# ==========================================================
# DISPLAY IMAGE
# ==========================================================

text = f"{emotion} ({confidence:.2%})"

cv2.putText(

    original,

    text,

    (20,40),

    cv2.FONT_HERSHEY_SIMPLEX,

    1,

    (0,255,0),

    2

)

cv2.imshow(

    "Prediction",

    original

)

cv2.waitKey(0)

cv2.destroyAllWindows()