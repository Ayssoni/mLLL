"""
=========================================================
Emotion Recognition Configuration
Dataset : FER2013 (7 Classes)
Model   : EfficientNetB0
=========================================================
"""

import os

# =====================================================
# PROJECT PATHS
# =====================================================

# Change this to your project root
PROJECT_ROOT = "/Users/aysoni/Documents/mLLL/staG4/tLfLC"

# Dataset paths
TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# Create folders if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =====================================================
# DATASET
# =====================================================

IMG_SIZE = 224

CHANNELS = 3

NUM_CLASSES = 7

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# =====================================================
# TRAINING
# =====================================================

BATCH_SIZE = 32

INITIAL_EPOCHS = 20

FINE_TUNE_EPOCHS = 15

TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

# =====================================================
# LEARNING RATES
# =====================================================

INITIAL_LR = 1e-3

FINE_TUNE_LR = 1e-5

# =====================================================
# MODEL
# =====================================================

BACKBONE = "EfficientNetB0"

FREEZE_ALL = True

UNFREEZE_LAST = 30

# =====================================================
# REGULARIZATION
# =====================================================

DROPOUT_1 = 0.5

DROPOUT_2 = 0.3

DENSE_1 = 512

DENSE_2 = 256

# =====================================================
# CALLBACKS
# =====================================================

EARLY_STOPPING_PATIENCE = 5

REDUCE_LR_PATIENCE = 3

REDUCE_FACTOR = 0.2

MIN_LR = 1e-7

# =====================================================
# FILE NAMES
# =====================================================

BEST_MODEL = os.path.join(
    MODEL_DIR,
    "best_emotion_model.keras"
)

FINAL_MODEL = os.path.join(
    MODEL_DIR,
    "best_emotion_model.keras"
)

TENSORBOARD_LOGS = LOG_DIR

# =====================================================
# RANDOMNESS
# =====================================================

SEED = 42

# =====================================================
# PREDICTION
# =====================================================

CONFIDENCE_THRESHOLD = 0.50

# =====================================================
# WEBCAM
# =====================================================

CAMERA_INDEX = 0

FACE_PADDING = 20