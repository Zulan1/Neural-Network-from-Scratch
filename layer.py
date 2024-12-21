import numpy as np

from activations import Activation
from tensor import Tensor
from utils import pad

class Layer:
    """Layer class for neural network"""
    def __init__(self,
                 in_shape: int, 
                 out_shape: int, 
                 activation: Activation, 
                 w_init: np.ndarray = None):
        """Initialize the layer
        Args:
            in_shape (int): input length of the layer
            out_shape (int): output length of the layer
            activation (Activation): activation function of the layer
            w_init (np.ndarray): initial weights of the layer
            solver (Solver): solver for the layer
        """
        self.activation = activation
        self.layers = [self]
        if not isinstance(self, ResNetLayer):
            if w_init:
                assert w_init.shape == (in_shape + 1, out_shape)
            self.W: Tensor = w_init if w_init else Tensor(np.random.randn(in_shape + 1, out_shape) * 0.01)
        self.A: np.ndarray = None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the layer
        Args:
            X (np.ndarray): input of the layer
        """
        self.A = self.activation(self.W.T @ pad(X))
        return self.A

    def backward(self, X: np.ndarray, layer, V) -> np.ndarray:
        """Backward pass of the layer
        Args:
            X (np.ndarray): input of the layer
            layer (Layer): layer object
            V (np.ndarray): gradient of the loss function
            activation (Activation): activation function of the layer
        """
        sigma_deriv = self.activation.grad(layer.A)
        grad_W = pad(X) @ (sigma_deriv * V).T
        grad_X = (layer.W @ (sigma_deriv * V))
        self.W.grad = grad_W
        return grad_X[:-1, :] # bias removed for next layer

class ResNetLayer(Layer): # X + W2 @ activation(W1 @ X)
    def __init__(self, in_shape, out_shape, activation: Activation, w_init = None, w_init2 = None):
        super().__init__(in_shape, out_shape, activation, w_init)
        if w_init:
            assert w_init.shape == (in_shape + 1, out_shape)
        if w_init2:
            assert w_init2.shape == (in_shape, out_shape)
        W1 = w_init if w_init else np.random.randn(in_shape + 1, out_shape) * 0.01
        W2 = w_init2 if w_init2 else np.random.randn(in_shape, out_shape) * 0.01
        self.W: Tensor = Tensor(np.stack([W1,
                                   pad(W2, axis=0, pad_value=0)],
                                   axis=2))
        self.inner_A: np.ndarray = None

    def forward(self, X: np.ndarray):
        W1 = self.W[:, :, 0] # k1 + 1 x k2
        W2 = self.W[:-1, :, 1] # k1 x k2
        out = self.activation(W1.T @ pad(X))
        self.inner_A = out
        out = X + W2 @ out
        self.A = out
        return out

    def backward(self, X: np.ndarray, layer: Layer, V): # dZ/dW2 = v @ Z_inner.T, dZ/dw1 = X @ (sigma'(W1 @ X) * (W2.T @ v)).T, dZ/X = (V.T + ((V.T @ W.2) * sigma'(W1 @ X).T) @ W1.T).T
        W1 = self.W[:, :, 0]
        W2 = self.W[:-1, :, 1]
        sigma_deriv = self.activation.grad(layer.inner_A) # k2 x m
        grad_W2 = V @ layer.inner_A.T
        grad_W1 = pad(X) @ (sigma_deriv * (W2.T @ V)).T # k1 + 1 x k2
        grad_X = (V.T + ((V.T @ W2) * sigma_deriv.T) @ W1[:-1, :].T).T # k1 x m
        self.W.grad = np.stack([
            grad_W1,
            pad(grad_W2, axis=0, pad_value=0)],
            axis=2)
        return grad_X

        

if __name__ == "__main__":
    pass
