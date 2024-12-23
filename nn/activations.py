import numpy as np

from abc import ABC, abstractmethod


class Activation(ABC):

    def __init__(self):
        pass

    def __call__(self, Z: np.ndarray):
        return self.activation(Z)

    @abstractmethod
    def activation(self, Z: np.ndarray) -> np.ndarray:
        """
        Applies an activation function to the input array.
        Parameters:
            A (np.ndarray): The input array to which the activation function will be applied.
        Returns:
            np.ndarray: The result of applying the activation function to the input array.
        """

        pass

    @abstractmethod
    def grad(self, A: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function.
        
        Returns:
            np.ndarray: Gradient of the activation function.
        """
        pass


class Tanh(Activation):
    def activation(self, Z: np.ndarray):
        """
        Perform the forward pass of the neural network layer.
        
        Returns:
            np.ndarray: Output of the layer after applying the tanh activation function.
        """
        return np.tanh(Z)

    def grad(self, A: np.ndarray):
        return (1 - A ** 2)


class ReLU(Activation):
    def activation(self, Z: np.ndarray):
        """
        Applies the ReLU (Rectified Linear Unit) activation function to the input array.
        The ReLU function sets all negative values in the input array to zero.
        Parameters:
        A (np.ndarray): Input array to which the ReLU activation function will be applied.
        Returns:
        np.ndarray: Output array with ReLU activation applied.
        """
        A = Z.copy()
        A[A < 0] = 0
        return A

    def grad(self, A: np.ndarray):
        """
        Compute the gradient of the activation function.
        Parameters:
        A (np.ndarray): The input array for which the gradient is to be computed.
        Returns:
        np.ndarray: The gradient of the activation function with respect to the input array.
        """

        return np.where(A, 1, 0)


class SoftMax(Activation):
    def activation(self, Z: np.ndarray):
        """
        Perform the forward pass of the neural network layer.
        
        Returns:
            np.ndarray: Output of the layer after applying the softmax activation function.
        """
        exps = np.exp(Z - np.max(Z, axis=0))
        return exps / np.sum(exps, axis=0)

    def grad(self, A: np.ndarray):
        return np.ones_like(A) # not true, but works for the purposes of softmax here.