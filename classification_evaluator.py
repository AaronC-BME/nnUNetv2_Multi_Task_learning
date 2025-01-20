import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class ClassificationEvaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        """
        Evaluate classification performance.

        Parameters:
        y_pred (list or np.ndarray): Predicted class labels.
        y_true (list or np.ndarray): Ground truth class labels.

        Returns:
        dict: A dictionary containing accuracy, precision, recall, f1_score,
              macro-average F1 score, and the confusion matrix.
        """
        # Ensure inputs are numpy arrays
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Return metrics in a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_f1_score": macro_f1
        }

    def evaluate_validation (self, y_true, y_pred):
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Return metrics in a dictionary
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "macro_f1_score": macro_f1,
            "confusion_matrix": conf_matrix
        }
