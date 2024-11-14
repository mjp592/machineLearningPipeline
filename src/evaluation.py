from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, precision_recall_curve, auc, matthews_corrcoef, classification_report
)
import numpy as np


class ModelEvaluator:
    def __init__(self, average='binary'):
        """
        Initialize the model evaluator with options for evaluation metrics.

        Parameters:
        - average (str): The averaging method for metrics that require it (e.g., precision, recall, f1).
                         Common options are 'binary', 'micro', 'macro', and 'weighted'.
        """
        self.average = average

    def evaluate(self, model, x_test, y_test):
        """
        Evaluate the provided model on the test set using common metrics.

        Parameters:
        - model: The trained model to evaluate.
        - X_test: Features of the test set.
        - y_test: Target labels of the test set.

        Returns:
        - metrics (dict): Dictionary of evaluation metrics, including accuracy, precision, recall, F1,
                          confusion matrix, ROC AUC (for binary), Precision-Recall AUC, MCC, and classification report.
        """
        # Generate predictions and prediction probabilities (if supported by model)
        predictions = model.predict(x_test)
        try:
            probabilities = model.predict_proba(x_test)[:, 1]
        except AttributeError:
            probabilities = None

        # Calculate basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average=self.average),
            'recall': recall_score(y_test, predictions, average=self.average),
            'f1_score': f1_score(y_test, predictions, average=self.average),
            'confusion_matrix': confusion_matrix(y_test, predictions),
            'classification_report': classification_report(y_test, predictions, output_dict=True),
            'mcc': matthews_corrcoef(y_test, predictions) if len(np.unique(y_test)) == 2 else None
        }

        # Add ROC AUC score for binary classification
        if probabilities is not None and len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, probabilities)

        # Add Precision-Recall AUC score for binary classification
        if probabilities is not None and len(np.unique(y_test)) == 2:
            precision, recall, _ = precision_recall_curve(y_test, probabilities)
            metrics['pr_auc'] = auc(recall, precision)

        return metrics
