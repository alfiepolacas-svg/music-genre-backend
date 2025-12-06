"""
Custom Metrics for Training
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

class MetricsCalculator:
    """Calculate various metrics for model evaluation"""
    
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)
    
    @staticmethod
    def calculate_confusion_matrix(y_true, y_pred, labels):
        """Calculate confusion matrix"""
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def calculate_classification_report(y_true, y_pred, target_names):
        """Generate classification report"""
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4
        )
    
    @staticmethod
    def calculate_per_class_metrics(y_true, y_pred):
        """Calculate precision, recall, f1 per class"""
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred
        )
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

metrics_calculator = MetricsCalculator()