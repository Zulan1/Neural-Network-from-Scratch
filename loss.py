import numpy as np

from abc import ABC, abstractmethod
from layer import ResNetLayer
from utils import pad

class Loss(ABC):
    def init(self):
        pass

    @abstractmethod
    def backward(self, model, C: np.ndarray, X: np.ndarray):
        raise NotImplementedError

class CrossEntropy(Loss):
    def __call__(self, A: np.ndarray, C: np.ndarray):
        """
        Perform the forward pass of the loss function.

        Returns:
            np.ndarray: The loss value.
        """
        epsilon = 1e-10
        m = A.shape[1]
        loss = -(1 / m) * np.sum(C * np.log(A + epsilon))
        return loss

    def backward(self, model, X: np.ndarray, C: np.ndarray):
        """
        Perform the backward pass of the loss function.

        Returns:
            np.ndarray: The gradient of the loss function with respect to the weights.
        """
        V = model.layers[-1].A - C # (Z - C) / Z * (1 - Z)
        for i in reversed(range(len(model.layers))):
            curr_layer = model.layers[i]
            curr_X = model.layers[i - 1].A if i > 0 else X
            V = curr_layer.backward(curr_X, curr_layer, V)
            
