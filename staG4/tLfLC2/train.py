"""
=========================================================
Emotion Recognition V2
Stage 1 Training (Feature Extraction)
=========================================================
"""

import tensorflow as tf

from dataset import get_datasets
from augmentations import apply_augmentation
from model import build_model
from losses import get_loss
from callbacks import get_callbacks
from scheduler import get_scheduler

import config


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = tf.keras.optimizers.AdamW(

    learning_rate=config.MAX_LEARNING_RATE,

    weight_decay=config.WEIGHT_DECAY

)


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("Emotion Recognition V2")
    print("Stage 1 : Feature Extraction")
    print("=" * 60)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_ds, valid_ds, test_ds = get_datasets()

    train_ds = apply_augmentation(train_ds)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model, backbone = build_model()

    # --------------------------------------------------
    # Compile
    # --------------------------------------------------

    model.compile(

        optimizer=optimizer,

        loss=get_loss(),

        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(

                name="precision"

            ),

            tf.keras.metrics.Recall(

                name="recall"

            )

        ]

    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    model.summary()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    history = model.fit(

        train_ds,

        validation_data=valid_ds,

        epochs=config.FEATURE_EXTRACTION_EPOCHS,

        callbacks=[

            *get_callbacks(),

            get_scheduler()

        ]

    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    model.save(

        config.FINAL_MODEL

    )

    print("\nModel Saved Successfully.")

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    print("\nEvaluating...\n")

    model.evaluate(

        test_ds

    )


if __name__ == "__main__":

    main()