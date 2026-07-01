"""
=========================================================
Emotion Recognition V2
Model Evaluation
=========================================================
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import config
from dataset import get_datasets


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_model():

    print("\nLoading Model...\n")

    model = tf.keras.models.load_model(

        config.FINAL_MODEL,

        compile=False

    )

    return model


# ==========================================================
# EVALUATE
# ==========================================================

def evaluate_model(model, test_ds):

    print("\nEvaluating...\n")

    results = model.evaluate(

        test_ds,

        verbose=1

    )

    print("\n====================================")

    print(f"Loss     : {results[0]:.4f}")

    print(f"Accuracy : {results[1]:.4f}")

    print("====================================")


# ==========================================================
# PREDICTIONS
# ==========================================================

def predict_dataset(model, test_ds):

    predictions = model.predict(

        test_ds,

        verbose=1

    )

    y_pred = np.argmax(

        predictions,

        axis=1

    )

    y_true = np.concatenate(

        [

            np.argmax(labels.numpy(), axis=1)

            for _, labels in test_ds

        ]

    )

    return y_true, y_pred


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

def print_report(y_true, y_pred):

    print("\nClassification Report\n")

    report = classification_report(

        y_true,

        y_pred,

        target_names=config.CLASS_NAMES

    )

    print(report)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(

        y_true,

        y_pred

    )

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=config.CLASS_NAMES

    )

    fig, ax = plt.subplots(figsize=(10,10))

    disp.plot(

        cmap="Blues",

        ax=ax,

        colorbar=False

    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(

        config.RESULT_DIR /

        "confusion_matrix.png"

    )

    plt.show()


# ==========================================================
# MAIN
# ==========================================================

def main():

    _, _, test_ds = get_datasets()

    model = load_model()

    evaluate_model(

        model,

        test_ds

    )

    y_true, y_pred = predict_dataset(

        model,

        test_ds

    )

    print_report(

        y_true,

        y_pred

    )

    plot_confusion_matrix(

        y_true,

        y_pred

    )


if __name__ == "__main__":

    main()