import numpy as np

def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the accuracy of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The accuracy of the model.
    """
    return np.mean(y_pred == y_true)

def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the precision of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The precision of the model.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp)

def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the recall of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The recall of the model.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)

def f1_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the F1 score of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The F1 score of the model.
    """
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return 2 * (prec * rec) / (prec + rec)