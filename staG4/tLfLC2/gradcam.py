"""
=========================================================
Emotion Recognition V2
Grad-CAM Visualization
=========================================================
"""

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import config


# ==========================================================
# IMAGE PATH
# ==========================================================

IMAGE_PATH = r"/Users/aysoni/Desktop/test.jpg"

# ==========================================================
# LOAD MODEL
# ==========================================================

model = tf.keras.models.load_model(

    config.FINAL_MODEL,

    compile=False

)


# ==========================================================
# LAST CONV LAYER
# ==========================================================

LAST_CONV_LAYER = "top_activation"

# If this layer is not found,
# run:
#
# for layer in model.layers:
#     print(layer.name)
#
# and replace it with the last Conv layer.


# ==========================================================
# LOAD IMAGE
# ==========================================================

image = cv2.imread(IMAGE_PATH)

image_rgb = cv2.cvtColor(

    image,

    cv2.COLOR_BGR2RGB

)

resized = cv2.resize(

    image_rgb,

    (

        config.IMAGE_SIZE,

        config.IMAGE_SIZE

    )

)

input_image = np.expand_dims(

    resized,

    axis=0

).astype(np.float32)


# ==========================================================
# GRAD MODEL
# ==========================================================

grad_model = tf.keras.models.Model(

    model.inputs,

    [

        model.get_layer(LAST_CONV_LAYER).output,

        model.output

    ]

)


# ==========================================================
# COMPUTE GRADIENT
# ==========================================================

with tf.GradientTape() as tape:

    conv_output, predictions = grad_model(input_image)

    class_index = tf.argmax(predictions[0])

    loss = predictions[:, class_index]

grads = tape.gradient(

    loss,

    conv_output

)

pooled_grads = tf.reduce_mean(

    grads,

    axis=(0,1,2)

)

conv_output = conv_output[0]

heatmap = conv_output @ pooled_grads[..., tf.newaxis]

heatmap = tf.squeeze(heatmap)

heatmap = tf.maximum(

    heatmap,

    0

)

heatmap /= tf.reduce_max(heatmap)

heatmap = heatmap.numpy()


# ==========================================================
# OVERLAY
# ==========================================================

heatmap = cv2.resize(

    heatmap,

    (

        image.shape[1],

        image.shape[0]

    )

)

heatmap = np.uint8(

    255 * heatmap

)

heatmap = cv2.applyColorMap(

    heatmap,

    cv2.COLORMAP_JET

)

overlay = cv2.addWeighted(

    image,

    0.6,

    heatmap,

    0.4,

    0

)


# ==========================================================
# RESULT
# ==========================================================

emotion = config.CLASS_NAMES[class_index]

confidence = predictions[0][class_index]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)

plt.imshow(image_rgb)

plt.title("Original")

plt.axis("off")

plt.subplot(1,2,2)

plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))

plt.title(

    f"{emotion} ({confidence:.2%})"

)

plt.axis("off")

plt.show()