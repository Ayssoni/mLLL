"""
=========================================================
Evaluate Emotion Recognition Model
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (

    confusion_matrix,

    classification_report,

    accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

import config

from dataset import get_datasets

# =====================================================
# LOAD DATASET
# =====================================================

print("\nLoading Dataset...\n")

_, test_dataset, _ = get_datasets()

print("Dataset Loaded\n")

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading Model...\n")

model = tf.keras.models.load_model(

    config.FINAL_MODEL

)

print("Model Loaded Successfully\n")

# =====================================================
# PREDICT
# =====================================================

print("Predicting...\n")

predictions = model.predict(

    test_dataset,

    verbose=1

)

# =====================================================
# TRUE LABELS
# =====================================================

y_true = test_dataset.classes

# =====================================================
# PREDICTED LABELS
# =====================================================

y_pred = np.argmax(

    predictions,

    axis=1

)

# =====================================================
# ACCURACY
# =====================================================

accuracy = accuracy_score(

    y_true,

    y_pred

)

precision = precision_score(

    y_true,

    y_pred,

    average="weighted"

)

recall = recall_score(

    y_true,

    y_pred,

    average="weighted"

)

f1 = f1_score(

    y_true,

    y_pred,

    average="weighted"

)

print("\n=====================================")
print("Evaluation")
print("=====================================\n")

print(f"Accuracy  : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall    : {recall:.4f}")

print(f"F1 Score  : {f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report = classification_report(

    y_true,

    y_pred,

    target_names=config.CLASS_NAMES

)

print("\n=====================================")
print("Classification Report")
print("=====================================\n")

print(report)

# =====================================================
# SAVE REPORT
# =====================================================

report_path = os.path.join(

    config.RESULT_DIR,

    "classification_report.txt"

)

with open(

    report_path,

    "w"

) as file:

    file.write(report)

print("\nClassification Report Saved")

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(

    y_true,

    y_pred

)

# =====================================================
# PLOT CONFUSION MATRIX
# =====================================================

plt.figure(figsize=(10,8))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=config.CLASS_NAMES,

    yticklabels=config.CLASS_NAMES

)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(

    os.path.join(

        config.RESULT_DIR,

        "confusion_matrix.png"

    )

)

plt.show()

print("\nConfusion Matrix Saved")

# =====================================================
# PER CLASS ACCURACY
# =====================================================

print("\n=====================================")
print("Per Class Accuracy")
print("=====================================\n")

for i, class_name in enumerate(config.CLASS_NAMES):

    total = np.sum(cm[i])

    correct = cm[i][i]

    acc = (correct / total) * 100

    print(

        f"{class_name:10s} : {acc:.2f}%"

    )

print("\nEvaluation Completed Successfully")