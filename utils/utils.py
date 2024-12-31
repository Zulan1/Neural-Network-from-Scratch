import numpy as np

from typing import List
from nn import NeuralNetwork
from nn.loss import CrossEntropy, Loss
from nn.optimizer import SGD, Optimizer

def nn_builder(
        net_shape: List[int],
        activation: str,
        resnet: bool,
        loss: str,
        optim: str,
        lr: float,
        momentum: float
        ) -> tuple[NeuralNetwork, Optimizer, Loss]:
    # Initialize the model
    if resnet:
        assert len(set(net_shape[1:-1])) == 1, "ResNet layers must have the same number of units in each hidden layer"
    model = NeuralNetwork()
    L = len(net_shape) - 1
    for i in range(L):
        input_dim, output_dim = net_shape[i], net_shape[i + 1]
        if i == L - 1:
            model.add_layer(input_dim, output_dim, 'softmax')
        elif i != 0:
            model.add_layer(input_dim, output_dim, activation, resnet)
        else:
            model.add_layer(input_dim, output_dim, activation)

    # Select the optimizer
    if optim == 'sgd':
        optimizer_fn = SGD(lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    # Select the loss function
    if loss == 'crossentropy':
        loss_fn = CrossEntropy()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    
    return model, optimizer_fn, loss_fn

def split_into_batches(X: np.ndarray, batch_size):
    """
    Splits a NumPy array into batches of size `batch_size`.
    Args:
        arr (np.ndarray): The input array.
        batch_size (int): Size of each batch.
    Returns:
        list: A list of NumPy arrays, each of size `batch_size` (except possibly the last one).
    """
    m = X.shape[1]
    return [X[:, i:i + batch_size] for i in range(0, m, batch_size)]

def train_test_split(X: np.ndarray, C: np.ndarray, test_size: float):
    """
    Splits the data into training and testing sets.
    Args:
        X (np.ndarray): The input data.
        C (np.ndarray): The target data.
        test_size (float): The fraction of the data to reserve for testing.
    Returns:
        tuple: A tuple containing the training and testing data.
    """
    m = X.shape[1]
    idx = np.random.permutation(m)
    split = int(m * (1 - test_size))
    X_train, X_test = X[:, idx[:split]], X[:, idx[split:]]
    C_train, C_test = C[:, idx[:split]], C[:, idx[split:]]
    return X_train, C_train, X_test, C_test
