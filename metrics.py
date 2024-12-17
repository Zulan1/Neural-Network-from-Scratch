import numpy as np

def accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the accuracy of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The accuracy of the model.
    """
    y_pred = np.argmax(probs, axis=0)
    return np.mean(y_pred == np.argmax(y_true, axis=0))

def precision(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the precision of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The precision of the model.
    """
    y_pred = np.argmax(probs, axis=0)
    y_true = np.argmax(y_true, axis=0)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp)

def recall(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the recall of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The recall of the model.
    """
    y_pred = np.argmax(probs, axis=0)
    y_true = np.argmax(y_true, axis=0)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn)

def f1_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Calculate the F1 score of the model.

    Parameters:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        float: The F1 score of the model.
    """
    y_pred = np.argmax(probs, axis=0)
    y_true = np.argmax(y_true, axis=0)
    prec = precision(y_pred, y_true)
    rec = recall(y_pred, y_true)
    return 2 * (prec * rec) / (prec + rec)