"""
=========================================================
Real Time Emotion Recognition
FER2013
EfficientNetB0
=========================================================
"""

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

import config

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading model...")

model = tf.keras.models.load_model(
    config.FINAL_MODEL
)

print("Model Loaded")

# =====================================================
# LOAD FACE DETECTOR
# =====================================================

face_detector = cv2.CascadeClassifier(

    cv2.data.haarcascades +

    "haarcascade_frontalface_default.xml"

)

# =====================================================
# OPEN CAMERA
# =====================================================

camera = cv2.VideoCapture(config.CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not camera.isOpened():

    raise RuntimeError("Unable to Open Camera")

print("Camera Started")

# =====================================================
# LOOP
# =====================================================

while True:

    success, frame = camera.read()

    if not success:

        break

    gray = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2GRAY

    )

    faces = face_detector.detectMultiScale(

        gray,

        scaleFactor=1.2,

        minNeighbors=5,

        minSize=(80,80)

    )

    for (x,y,w,h) in faces:

        face = frame[

            y:y+h,

            x:x+w

        ]

        rgb = cv2.cvtColor(

            face,

            cv2.COLOR_BGR2RGB

        )

        rgb = cv2.resize(

            rgb,

            (

                config.IMG_SIZE,

                config.IMG_SIZE

            )

        )

        rgb = rgb.astype("float32")

        rgb = preprocess_input(rgb)

        rgb = np.expand_dims(

            rgb,

            axis=0

        )

        prediction = model.predict(

            rgb,

            verbose=0

        )[0]

        index = np.argmax(prediction)

        emotion = config.CLASS_NAMES[index]

        confidence = prediction[index] * 100

        cv2.rectangle(

            frame,

            (x,y),

            (x+w,y+h),

            (0,255,0),

            2

        )

        label = f"{emotion} {confidence:.1f}%"

        cv2.putText(

            frame,

            label,

            (x,y-10),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0,255,0),

            2

        )

    cv2.imshow(

        "Emotion Recognition",

        frame

    )

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):

        break

# =====================================================
# RELEASE
# =====================================================

camera.release()

cv2.destroyAllWindows()