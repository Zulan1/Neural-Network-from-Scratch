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

# def pad(X: np.ndarray) -> np.ndarray:
#     """
#     Pad the input data with ones.

#     Parameters:
#         X (np.ndarray): The input data.

#     Returns:
#         np.ndarray: The input data with ones.
#     """
#     m = X.shape[1]
#     return np.concatenate((X, np.ones((1, m))), axis=0)

def pad(X, axis=0, pad_width=1, pad_value=1):
    """
    Pads a numpy array along a specified axis.
    
    Parameters:
        X (numpy.ndarray): Input array.
        axis (int): The axis to pad.
        pad_width (int): The amount of padding to add after the axis. Default is 1.
        pad_value (int): The value to pad with. Default is 0.
        
    Returns:
        numpy.ndarray: Padded array.
    """
    # Create a tuple of (before, after) for each axis
    pad_config = [(0, 0)] * X.ndim  # No padding for all axes initially
    pad_config[axis] = (0, pad_width)  # Add padding only to the specified axis

    # Apply padding
    padded_X = np.pad(X, pad_width=pad_config, mode='constant', constant_values=pad_value)
    return padded_X

if __name__ == "__main__": #Dvir
    pass