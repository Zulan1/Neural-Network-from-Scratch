from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class Activation(ABC):

    def __init__(self):
        pass

    def __call__(self, W: np.ndarray, X: np.ndarray):
        return self.activation(W, X)

    @abstractmethod
    def activation(self, A: np.ndarray) -> np.ndarray:
        """
        Applies an activation function to the input array.
        Parameters:
            A (np.ndarray): The input array to which the activation function will be applied.
        Returns:
            np.ndarray: The result of applying the activation function to the input array.
        """

        pass

    @abstractmethod
    def grad(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function.
        
        Returns:
            np.ndarray: Gradient of the activation function.
        """
        pass


class Tanh(Activation):
    def activation(self, A: np.ndarray):
        """
        Perform the forward pass of the neural network layer.
        
        Returns:
            np.ndarray: Output of the layer after applying the tanh activation function.
        """
        return np.tanh(A)

    def grad(self, Z: np.ndarray):
        return (1 - Z ** 2)


class ReLU(Activation):
    def activation(self, A: np.ndarray):
        """
        Applies the ReLU (Rectified Linear Unit) activation function to the input array.
        The ReLU function sets all negative values in the input array to zero.
        Parameters:
        A (np.ndarray): Input array to which the ReLU activation function will be applied.
        Returns:
        np.ndarray: Output array with ReLU activation applied.
        """
        Z = A.copy()
        Z[Z < 0] = 0
        return Z

    def grad(self, Z: np.ndarray):
        """
        Compute the gradient of the activation function.
        Parameters:
        A (np.ndarray): The input array for which the gradient is to be computed.
        Returns:
        np.ndarray: The gradient of the activation function with respect to the input array.
        """

        return np.where(Z, 1, 0)
