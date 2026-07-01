"""
=========================================================
Emotion Recognition V2
Webcam Emotion Recognition
=========================================================
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

import config

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Model...")

model = tf.keras.models.load_model(
    config.FINAL_MODEL,
    compile=False
)

print("Model Loaded")

# ==========================================================
# MEDIAPIPE
# ==========================================================

mp_face = mp.solutions.face_detection

detector = mp_face.FaceDetection(

    model_selection=1,

    min_detection_confidence=0.6

)

# ==========================================================
# CAMERA
# ==========================================================

camera = cv2.VideoCapture(config.CAMERA_INDEX)

if not camera.isOpened():

    raise RuntimeError("Camera not found.")

# ==========================================================
# LOOP
# ==========================================================

while True:

    success, frame = camera.read()

    if not success:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = detector.process(rgb)

    if results.detections:

        h, w, _ = frame.shape

        for detection in results.detections:

            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * w)

            y = int(box.ymin * h)

            bw = int(box.width * w)

            bh = int(box.height * h)

            x = max(0, x)
            y = max(0, y)

            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face_rgb = cv2.resize(
                face_rgb,
                (
                    config.IMAGE_SIZE,
                    config.IMAGE_SIZE
                )
            )

            face_rgb = face_rgb.astype(np.float32)

            face_rgb = np.expand_dims(
                face_rgb,
                axis=0
            )

            prediction = model.predict(
                face_rgb,
                verbose=0
            )

            index = np.argmax(prediction)

            confidence = prediction[0][index]

            emotion = config.CLASS_NAMES[index]

            cv2.rectangle(

                frame,

                (x, y),

                (x+bw, y+bh),

                (0,255,0),

                2

            )

            text = f"{emotion} {confidence:.2%}"

            cv2.putText(

                frame,

                text,

                (x, y-10),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0,255,0),

                2

            )

    cv2.imshow(

        "Emotion Recognition",

        frame

    )

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

camera.release()

cv2.destroyAllWindows()