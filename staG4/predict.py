import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("emotion_stage4.h5")

classes = ["angry", "happy", "sad"]

image_path = "/Users/aysoni/Documents/mLLL/dataset/train/happy/Training_71811.jpg"

print("Exists:", os.path.exists(image_path))

img = cv2.imread(image_path)

if img is None:
    print("Could not load image")
    exit()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

index = np.argmax(prediction)

print("Emotion:", classes[index])
print("Confidence:", np.max(prediction))