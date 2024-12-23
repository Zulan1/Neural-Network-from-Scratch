import numpy as np


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
    pad_config = [(0, 0)] * X.ndim
    pad_config[axis] = (0, pad_width)

    # Apply padding
    padded_X = np.pad(X, pad_width=pad_config, mode='constant', constant_values=pad_value)
    return padded_X

if __name__ == "__main__":
    pass