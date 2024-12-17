import numpy as np

from abc import ABC, abstractmethod

class Loss(ABC):
    def init(self):
        pass

    @abstractmethod
    def backward(self, model, C: np.ndarray, X: np.ndarray):
        raise NotImplementedError

class CrossEntropy(Loss):
    def __call__(self, Z: np.ndarray, C: np.ndarray):
        """
        Perform the forward pass of the loss function.

        Returns:
            np.ndarray: The loss value.
        """
        epsilon = 1e-10
        loss = -np.sum(C * np.log(Z + epsilon))
        return loss

    def backward(self, model, X: np.ndarray, C: np.ndarray):
        """
        Perform the backward pass of the loss function.

        Returns:
            np.ndarray: The gradient of the loss function with respect to the weights.
        """
        m = X.shape[1]
        V = model.layers[-1].Z - C
        for i in reversed(range(len(model.layers))):
            curr_layer = model.layers[i]
            curr_X = model.layers[i - 1].Z if i > 0 else X
            curr_X = np.concatenate((curr_X, np.ones((1, m))), axis=0) # add bias
            sigma_deriv = curr_layer.activation.grad(curr_layer.Z) 
            grad_X = curr_layer.W @ (sigma_deriv * V)
            grad_W = curr_X @ (sigma_deriv * V).T
            V = grad_X[:-1, :] # remove bias
            curr_layer.W.grad = grad_W



    # def grad_W(self, _ : np.ndarray, X: np.ndarray, Z: np.ndarray, C: np.ndarray):
    #     """
    #     Perform the backward pass of the loss function.

    #     Returns:
    #         np.ndarray: The gradient of the loss function with respect to the weights.
    #     """
    #     m: int = Z.shape[0]
    #     dW: np.ndarray = X @ (Z - C)
    #     return dW

    # def gradX(self, W: np.ndarray, : np.ndarray, Z: np.ndarray, C: np.ndarray):
    #     """
    #     Perform the backward pass of the loss function.

    #     Returns:
    #         np.ndarray: The gradient of the loss function with respect to the weights.
    #     """
    #     m: int = Z.shape[0]
    #     dX: np.ndarray = W @ (Z - C).T
    #     return dX