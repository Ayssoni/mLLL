import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 15

train_generator = ImageDataGenerator(
    rescale=1.0/255
)

test_generator = ImageDataGenerator(
    rescale=1.0/255
)

train_data = train_generator.flow_from_directory(
    "/Users/aysoni/Documents/mLLL/dataSET1/train",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_data = test_generator.flow_from_directory(
    "/Users/aysoni/Documents/mLLL/dataSET1/test",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("Class Mapping:")
print(train_data.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation="relu",
           input_shape=(48,48,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128, activation="relu"),

    Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=EPOCHS
)

model.save("emotion_model.h5")

print("Model Saved Successfully")