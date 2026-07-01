"""
=========================================================
Emotion Recognition V2
Callbacks
=========================================================
"""

import tensorflow as tf

from tensorflow.keras.callbacks import (

    ModelCheckpoint,

    EarlyStopping,

    ReduceLROnPlateau,

    TensorBoard,

    CSVLogger

)

import config


# ==========================================================
# CHECKPOINT
# ==========================================================

checkpoint = ModelCheckpoint(

    filepath=config.BEST_MODEL,

    monitor=config.MONITOR,

    save_best_only=config.SAVE_BEST_ONLY,

    save_weights_only=config.SAVE_WEIGHTS_ONLY,

    mode=config.MODE,

    verbose=config.VERBOSE

)


# ==========================================================
# EARLY STOPPING
# ==========================================================

early_stopping = EarlyStopping(

    monitor=config.MONITOR,

    patience=config.EARLY_STOPPING_PATIENCE,

    restore_best_weights=True,

    mode=config.MODE,

    verbose=config.VERBOSE

)


# ==========================================================
# REDUCE LEARNING RATE
# ==========================================================

reduce_lr = ReduceLROnPlateau(

    monitor=config.MONITOR,

    factor=config.REDUCE_FACTOR,

    patience=config.REDUCE_LR_PATIENCE,

    min_lr=config.MIN_LR,

    mode=config.MODE,

    verbose=config.VERBOSE

)


# ==========================================================
# TENSORBOARD
# ==========================================================

tensorboard = TensorBoard(

    log_dir=config.TENSORBOARD_LOG,

    histogram_freq=1,

    write_graph=True,

    write_images=False

)


# ==========================================================
# CSV LOGGER
# ==========================================================

csv_logger = CSVLogger(

    config.CSV_LOG,

    append=False

)


# ==========================================================
# RETURN CALLBACKS
# ==========================================================

def get_callbacks():

    return [

        checkpoint,

        early_stopping,

        reduce_lr,

        tensorboard,

        csv_logger

    ]


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    callbacks = get_callbacks()

    print("\nCallbacks Loaded\n")

    for cb in callbacks:

        print(type(cb).__name__)