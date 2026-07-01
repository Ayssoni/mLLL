"""
=========================================================
Emotion Recognition V2
Stage 2 - Progressive Fine Tuning
=========================================================
"""

import tensorflow as tf

import config

from dataset import get_datasets
from augmentations import apply_augmentation
from model import build_model, unfreeze_backbone
from losses import get_loss
from callbacks import get_callbacks
from scheduler import get_scheduler


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    print("\nLoading Best Model...\n")

    model = tf.keras.models.load_model(

        config.CHECKPOINT_FILE,

        compile=False

    )

    return model


# ==========================================================
# FIND BACKBONE
# ==========================================================

def get_backbone(model):

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            return layer

    return None


# ==========================================================
# FINE TUNE
# ==========================================================

def fine_tune(

        model,

        backbone,

        train_ds,

        valid_ds,

        unfreeze_layers,

        epochs

):

    print("\n" + "=" * 60)

    print(

        f"Fine Tuning Last {unfreeze_layers} Layers"

    )

    print("=" * 60)

    unfreeze_backbone(

        backbone,

        unfreeze_layers

    )

    optimizer = tf.keras.optimizers.AdamW(

        learning_rate=config.FINE_TUNE_LR,

        weight_decay=config.WEIGHT_DECAY

    )

    model.compile(

        optimizer=optimizer,

        loss=get_loss(),

        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(),

            tf.keras.metrics.Recall()

        ]

    )

    history = model.fit(

        train_ds,

        validation_data=valid_ds,

        epochs=epochs,

        callbacks=[

            *get_callbacks(),

            get_scheduler()

        ]

    )

    return history


# ==========================================================
# MAIN
# ==========================================================

def main():

    train_ds, valid_ds, test_ds = get_datasets()

    train_ds = apply_augmentation(train_ds)

    model = load_model()

    backbone = get_backbone(model)

    if backbone is None:

        raise RuntimeError(

            "Backbone not found."

        )

    # ----------------------------------------
    # Stage 1
    # ----------------------------------------

    fine_tune(

        model,

        backbone,

        train_ds,

        valid_ds,

        unfreeze_layers=20,

        epochs=config.FINE_TUNE_STAGE1

    )

    # ----------------------------------------
    # Stage 2
    # ----------------------------------------

    fine_tune(

        model,

        backbone,

        train_ds,

        valid_ds,

        unfreeze_layers=60,

        epochs=config.FINE_TUNE_STAGE2

    )

    # ----------------------------------------
    # Stage 3
    # ----------------------------------------

    fine_tune(

        model,

        backbone,

        train_ds,

        valid_ds,

        unfreeze_layers=len(backbone.layers),

        epochs=config.FINE_TUNE_STAGE3

    )

    print("\nEvaluating...\n")

    model.evaluate(test_ds)

    model.save(

        config.FINAL_MODEL

    )

    print("\nFine Tuned Model Saved Successfully.")


if __name__ == "__main__":

    main()