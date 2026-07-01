"""
=========================================================
Emotion Recognition V2
Loss Functions
=========================================================
"""

import tensorflow as tf
import config


# ==========================================================
# CROSS ENTROPY
# ==========================================================

def cross_entropy():

    return tf.keras.losses.CategoricalCrossentropy()


# ==========================================================
# LABEL SMOOTHING
# ==========================================================

def label_smoothing():

    return tf.keras.losses.CategoricalCrossentropy(

        label_smoothing=config.LABEL_SMOOTHING

    )


# ==========================================================
# FOCAL LOSS
# ==========================================================

class FocalLoss(tf.keras.losses.Loss):

    def __init__(

        self,

        alpha=config.FOCAL_ALPHA,

        gamma=config.FOCAL_GAMMA,

        name="FocalLoss"

    ):

        super().__init__(name=name)

        self.alpha = alpha

        self.gamma = gamma

    def call(self, y_true, y_pred):

        y_pred = tf.clip_by_value(

            y_pred,

            1e-7,

            1.0 - 1e-7

        )

        ce = -y_true * tf.math.log(y_pred)

        weight = self.alpha * tf.pow(

            1 - y_pred,

            self.gamma

        )

        loss = weight * ce

        return tf.reduce_mean(

            tf.reduce_sum(loss, axis=1)

        )


# ==========================================================
# GET LOSS
# ==========================================================

def get_loss():

    if config.LOSS_FUNCTION == "cross_entropy":

        print("Using Cross Entropy Loss")

        return cross_entropy()

    elif config.LOSS_FUNCTION == "label_smoothing":

        print("Using Label Smoothing Loss")

        return label_smoothing()

    elif config.LOSS_FUNCTION == "focal":

        print("Using Focal Loss")

        return FocalLoss()

    else:

        raise ValueError(

            "Unknown Loss Function"

        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(get_loss())