import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("emotion_model_stage3.h5")

# Update according to your class_indices output
classes = [
    "angry",
    "happy",
    "sad"
]

# Load image
img_path = "/Users/aysoni/Documents/mLLL/dataset/train/angry/Training_3908.jpg"

img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (48,48))
img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)

index = np.argmax(prediction)

emotion = classes[index]

print("\nPredicted Emotion:", emotion)
print("Confidence:", round(np.max(prediction)*100,2), "%")