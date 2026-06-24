import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# We'll use CIFAR-10 — 60,000 color images, 10 classes
# airplane, car, bird, cat, deer, dog, frog, horse, ship, truck
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Class names
class_names = ['airplane','car','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("X_train shape:", X_train.shape)   # (50000, 32, 32, 3)
print("X_test shape :", X_test.shape)    # (10000, 32, 32, 3)

# Visualize first 10 images
plt.figure(figsize=(12,3))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]], fontsize=7)
    plt.axis('off')
plt.suptitle("CIFAR-10 Sample Images")
plt.show()