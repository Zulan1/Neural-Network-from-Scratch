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

        def calc_grads(X, Z, W, V, activation, residual=False):
            X = pad(X)
            sigma_deriv = activation.grad(Z)
            grad_X = W @ (sigma_deriv * V)
            grad_W = X @ (sigma_deriv * V).T
            V = grad_X[:-1, :]
            return grad_W, V

        for i in reversed(range(len(model.layers))):
            # curr_layer = model.layers[i]
            # curr_X = model.layers[i - 1].Z if i > 0 else X
            # curr_X = np.concatenate((curr_X, np.ones((1, m))), axis=0) # add bias
            # sigma_deriv = curr_layer.activation.grad(curr_layer.Z) 
            # grad_X = curr_layer.W @ (sigma_deriv * V)
            # grad_W = curr_X @ (sigma_deriv * V).T
            # V = grad_X[:-1, :] # remove bias
            # curr_layer.W.grad = grad_W
            curr_layer = model.layers[i]
            curr_X = model.layers[i - 1].Z if i > 0 else X
            curr_W = curr_layer.W if not isinstance(curr_layer, ResNetLayer) else curr_layer.W[:, :, 1]
            curr_Z = curr_layer.Z
            curr_activation = curr_layer.activation
            grad_W, V = calc_grads(curr_X, curr_Z, curr_W, V, curr_activation)
            if isinstance(curr_layer, ResNetLayer):
                curr_W = curr_layer.W[:, :, 0]
                curr_inner_Z = curr_layer.inner_Z
                inner_grad_W, V = calc_grads(curr_X, curr_inner_Z, curr_W, V, curr_activation)
                grad_W = np.dstack((inner_grad_W, grad_W))
            
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