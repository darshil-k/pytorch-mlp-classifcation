from enum import Enum


class ProblemType(Enum):
    """
    This class is used to define the problem type.
    """
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class AccuracyTypes(Enum):
    """
    This class is used to define the accuracy types.
    """
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1-score"
    RMSE = "Root Mean Squared Error"
    MAE = "Mean Absolute Error"

