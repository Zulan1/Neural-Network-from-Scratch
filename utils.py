import numpy as np

def train_test_split(X: np.ndarray, Y: np.ndarray, test_size: float = 0.2):
    """
    Split the data into training and testing sets.

    Parameters:
        X (np.ndarray): The input data.
        Y (np.ndarray): The target data.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training and testing sets.
    """
    m = X.shape[1]
    split = int((1 - test_size) * m)
    indices = np.random.permutation(m)
    training_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[:, training_idx], X[:, test_idx]
    Y_train, Y_test = Y[:, training_idx], Y[:, test_idx]
    return X_train, X_test, Y_train, Y_test

if __name__ == "__main__": #Dvir
    pass