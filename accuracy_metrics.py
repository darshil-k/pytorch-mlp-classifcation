"""
This file is used to define the accuracy metrics for the model.
"""
from enum import Enum
from typing import List, Dict, Any
from enums import ProblemType, AccuracyTypes
from torchmetrics import Accuracy, F1Score, Recall, Precision







class AccuracyMetrics:
    """
    This class is used to define the accuracy metrics for the model.
    """
    def __init__(self, problem_type: str, device: str):
        """
        This method initializes the class.
        :param problem_type: The problem type.
        """
        self.problem_to_accuracy_mapping = {
                ProblemType.REGRESSION.value : [AccuracyTypes.RMSE.value, AccuracyTypes.MAE.value],
                ProblemType.CLASSIFICATION.value : [AccuracyTypes.ACCURACY.value, AccuracyTypes.PRECISION.value, AccuracyTypes.RECALL.value, AccuracyTypes.F1_SCORE.value]
                }
        self.problem_type = problem_type
        self.accuracy_types = self.problem_to_accuracy_mapping[problem_type]
        self.device = device

    def calculate_accuracy(self, y_pred: Any, y_true: Any) -> Dict[str, float]:
        """
        This method calculates the accuracy.
        :param y_pred: The predicted labels.
        :param y_true: The true labels.
        :return: The accuracy for all metrics in the problem type.
        """

        if self.problem_type == ProblemType.CLASSIFICATION.value:
            accuracy_func = Accuracy(task="multiclass", num_classes=10, average="macro").to(self.device)
            f1_score_func = F1Score(task="multiclass", num_classes=10, average="macro").to(self.device)
            precision_func = Precision(task="multiclass", num_classes=10, average="macro").to(self.device)
            recall_func = Recall(task="multiclass", num_classes=10, average="macro").to(self.device)

            accuracies = {AccuracyTypes.ACCURACY.value: accuracy_func(y_pred, y_true).item(),
                          AccuracyTypes.PRECISION.value: precision_func(y_pred, y_true).item(),
                          AccuracyTypes.RECALL.value: recall_func(y_pred, y_true).item(),
                          AccuracyTypes.F1_SCORE.value: f1_score_func(y_pred, y_true).item()}
            return accuracies

        elif self.problem_type == ProblemType.REGRESSION.value:
            #TODO: Implement regression metrics
            pass
        else:
            raise ValueError("Invalid problem type")
