"""
=========================================================
Emotion Recognition V2
Configuration File
=========================================================
"""

from pathlib import Path

# ==========================================================
# PROJECT
# ==========================================================

PROJECT_NAME = "Emotion Recognition V2"

VERSION = "2.0"

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================================
# RAW DATASETS
# ==========================================================

RAW_DATASET = "/Users/aysoni/Documents/mLLL/dataset"

FER_PATH = "/Users/aysoni/Documents/mLLL/dataset/fer2013"

RAF_PATH = "/Users/aysoni/Documents/mLLL/dataset/rafdb"

AFFECTNET_PATH = "/Users/aysoni/Documents/mLLL/dataset/affectnet"

# ==========================================================
# PROCESSED DATASETS
# ==========================================================

PROCESSED_DATASET = PROJECT_ROOT / "processed_dataset"

FER_OUTPUT = PROCESSED_DATASET / "FER"

RAF_OUTPUT = PROCESSED_DATASET / "RAF"

AFFECT_OUTPUT = PROCESSED_DATASET / "AFFECT"

FINAL_DATASET = PROJECT_ROOT / "final_dataset"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

MODEL_DIR = PROJECT_ROOT / "models"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

LOG_DIR = PROJECT_ROOT / "logs"

RESULT_DIR = PROJECT_ROOT / "results"

GRAPH_DIR = PROJECT_ROOT / "graphs"

PREDICTION_DIR = PROJECT_ROOT / "predictions"

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

DIRECTORIES = [

    PROCESSED_DATASET,

    FER_OUTPUT,

    RAF_OUTPUT,

    AFFECT_OUTPUT,

    FINAL_DATASET,

    MODEL_DIR,

    CHECKPOINT_DIR,

    LOG_DIR,

    RESULT_DIR,

    GRAPH_DIR,

    PREDICTION_DIR

]

for directory in DIRECTORIES:

    directory.mkdir(

        parents=True,

        exist_ok=True

    )

# ==========================================================
# DATASET OPTIONS
# ==========================================================

USE_FER = True

USE_RAF = True

USE_AFFECT = True

# ==========================================================
# DATASET SAMPLING
# ==========================================================

FER_RATIO = 1.0

RAF_RATIO = 1.0

AFFECT_RATIO = 0.50

# ==========================================================
# SPLITS
# ==========================================================

TRAIN_FOLDER = "train"

VALID_FOLDER = "valid"

TEST_FOLDER = "test"

# ==========================================================
# EMOTION CLASSES
# ==========================================================

CLASS_NAMES = [

    "angry",

    "disgust",

    "fear",

    "happy",

    "neutral",

    "sad",

    "surprise"

]

NUM_CLASSES = len(CLASS_NAMES)

# ==========================================================
# RAF LABEL MAP
# ==========================================================

RAF_LABELS = {

    "1": "surprise",

    "2": "fear",

    "3": "disgust",

    "4": "happy",

    "5": "sad",

    "6": "angry",

    "7": "neutral"

}

# ==========================================================
# AFFECTNET LABEL MAP
# ==========================================================

# IMPORTANT:
# Verify this with your data.yaml.
# Change it if the class order is different.

AFFECT_LABELS = {

    0: "angry",

    1: "disgust",

    2: "fear",

    3: "happy",

    4: "neutral",

    5: "sad",

    6: "surprise"

}

# ==========================================================
# IMAGE SETTINGS
# ==========================================================

IMAGE_SIZE = 224

IMAGE_CHANNELS = 3

IMAGE_SHAPE = (

    IMAGE_SIZE,

    IMAGE_SIZE,

    IMAGE_CHANNELS

)

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 32

AUTOTUNE = -1

SHUFFLE_BUFFER = 1000

SEED = 42

# ==========================================================
# EPOCHS
# ==========================================================

FEATURE_EXTRACTION_EPOCHS = 15

FINE_TUNE_STAGE1 = 10

FINE_TUNE_STAGE2 = 10

FINE_TUNE_STAGE3 = 10

TOTAL_EPOCHS = (

    FEATURE_EXTRACTION_EPOCHS +

    FINE_TUNE_STAGE1 +

    FINE_TUNE_STAGE2 +

    FINE_TUNE_STAGE3

)

# ==========================================================
# LEARNING RATE
# ==========================================================

INITIAL_LR = 1e-3

FINE_TUNE_LR = 1e-5

MIN_LR = 1e-7

WEIGHT_DECAY = 1e-4

# ==========================================================
# MODEL
# ==========================================================

BACKBONE = "EfficientNetV2S"

DROPOUT_1 = 0.50

DROPOUT_2 = 0.30

DENSE_1 = 512

DENSE_2 = 256

LABEL_SMOOTHING = 0.1

# ==========================================================
# AUGMENTATION
# ==========================================================

USE_AUGMENTATION = True

ROTATION_FACTOR = 0.10

ZOOM_FACTOR = 0.15

CONTRAST_FACTOR = 0.10

BRIGHTNESS_FACTOR = 0.10

TRANSLATION_FACTOR = 0.10

# ==========================================================
# CHECKPOINT FILES
# ==========================================================

BEST_MODEL = MODEL_DIR / "best_model.keras"

FINAL_MODEL = MODEL_DIR / "emotion_v2.keras"

LAST_MODEL = MODEL_DIR / "last_model.keras"

# ==========================================================
# LOGGING
# ==========================================================

TENSORBOARD_LOG = LOG_DIR / "tensorboard"

CSV_LOG = LOG_DIR / "training_log.csv"

# ==========================================================
# EARLY STOPPING
# ==========================================================

EARLY_STOPPING_PATIENCE = 8

REDUCE_LR_PATIENCE = 4

REDUCE_FACTOR = 0.2

# ==========================================================
# CAMERA
# ==========================================================

CAMERA_INDEX = 0

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# PRINT CONFIG
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)

    print(PROJECT_NAME)

    print("=" * 60)

    print(f"Project Root : {PROJECT_ROOT}")

    print(f"FER2013      : {FER_PATH}")

    print(f"RAF-DB       : {RAF_PATH}")

    print(f"AffectNet    : {AFFECTNET_PATH}")

    print(f"Final Data   : {FINAL_DATASET}")

    print(f"Backbone     : {BACKBONE}")

    print(f"Image Size   : {IMAGE_SIZE}")

    print(f"Batch Size   : {BATCH_SIZE}")

    print(f"Classes      : {CLASS_NAMES}")

    print("=" * 60)

    # ==========================================================
    # LOSS FUNCTION
    # ==========================================================

    LOSS_FUNCTION = "label_smoothing"
    # Options:
    # "cross_entropy"
    # "label_smoothing"
    # "focal"

    FOCAL_ALPHA = 0.25
    FOCAL_GAMMA = 2.0

    LABEL_SMOOTHING = 0.1

    # ==========================================================
    # CALLBACK SETTINGS
    # ==========================================================

    MONITOR = "val_accuracy"

    SAVE_BEST_ONLY = True

    SAVE_WEIGHTS_ONLY = False

    MODE = "max"

    VERBOSE = 1

    EARLY_STOPPING_PATIENCE = 8

    REDUCE_LR_PATIENCE = 3

    REDUCE_FACTOR = 0.2

    MIN_LR = 1e-7

    # ==========================================================
    # LEARNING RATE SCHEDULER
    # ==========================================================

    USE_COSINE_SCHEDULER = True

    WARMUP_EPOCHS = 5

    INITIAL_LEARNING_RATE = 1e-5

    MAX_LEARNING_RATE = 1e-3

    MIN_LEARNING_RATE = 1e-6