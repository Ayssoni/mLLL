"""
=========================================================
Emotion Recognition V2
Learning Rate Scheduler
=========================================================
"""

import math

import tensorflow as tf

import config


# ==========================================================
# COSINE DECAY WITH WARMUP
# ==========================================================

def cosine_scheduler(epoch):

    # -----------------------------
    # Warmup
    # -----------------------------

    if epoch < config.WARMUP_EPOCHS:

        lr = (

            config.INITIAL_LEARNING_RATE +

            (

                config.MAX_LEARNING_RATE -

                config.INITIAL_LEARNING_RATE

            )

            *

            (

                epoch /

                config.WARMUP_EPOCHS

            )

        )

        return lr

    # -----------------------------
    # Cosine Decay
    # -----------------------------

    progress = (

        epoch -

        config.WARMUP_EPOCHS

    ) / (

        config.TOTAL_EPOCHS -

        config.WARMUP_EPOCHS

    )

    cosine_decay = 0.5 * (

        1 +

        math.cos(

            math.pi * progress

        )

    )

    lr = (

        config.MIN_LEARNING_RATE +

        (

            config.MAX_LEARNING_RATE -

            config.MIN_LEARNING_RATE

        )

        *

        cosine_decay

    )

    return lr


# ==========================================================
# CALLBACK
# ==========================================================

def get_scheduler():

    return tf.keras.callbacks.LearningRateScheduler(

        cosine_scheduler,

        verbose=1

    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print("\nLearning Rate Schedule\n")

    for epoch in range(config.TOTAL_EPOCHS):

        lr = cosine_scheduler(epoch)

        print(

            f"Epoch {epoch+1:02d} : {lr:.8f}"

        )