import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):

    @staticmethod
    @abstractmethod
    def activation(Z: np.ndarray) -> np.ndarray:
        """
        Applies an activation function to the input array.

        Parameters:
            Z (np.ndarray): The input array to which the activation function will be applied.

        Returns:
            np.ndarray: The result of applying the activation function to the input array.
        """
        pass

    @staticmethod
    @abstractmethod
    def grad(A: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the activation function.

        Parameters:
            A (np.ndarray): The input array for which the gradient is to be computed.

        Returns:
            np.ndarray: Gradient of the activation function.
        """
        pass

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        return self.activation(Z)


class Tanh(Activation):

    @staticmethod
    def activation(Z: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass using the tanh activation function.

        Parameters:
            Z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying the tanh activation function.
        """
        return np.tanh(Z)

    @staticmethod
    def grad(A: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the tanh activation function.

        Parameters:
            A (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the tanh activation function.
        """
        return 1 - A ** 2

class ReLU(Activation):

    @staticmethod
    def activation(Z: np.ndarray) -> np.ndarray:
        """
        Applies the ReLU activation function to the input array.

        Parameters:
            Z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying ReLU activation.
        """
        A = Z.copy()
        A[A < 0] = 0
        return A

    @staticmethod
    def grad(A: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the ReLU activation function.

        Parameters:
            A (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the ReLU activation function.
        """
        return np.where(A > 0, 1, 0)

class SoftMax(Activation):

    @staticmethod
    def activation(Z: np.ndarray) -> np.ndarray:
        """
        Applies the softmax activation function to the input array.

        Parameters:
            Z (np.ndarray): Input array.

        Returns:
            np.ndarray: Output array after applying softmax activation.
        """
        exps = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exps / np.sum(exps, axis=0, keepdims=True)

    @staticmethod
    def grad(A: np.ndarray) -> np.ndarray:
        """
        Placeholder gradient for the softmax activation function.

        Parameters:
            A (np.ndarray): Input array.

        Returns:
            np.ndarray: Gradient of the softmax activation function.
        """
        return np.ones_like(A)  # Placeholder; softmax gradients depend on the full input array.
