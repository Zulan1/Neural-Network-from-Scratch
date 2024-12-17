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
        if w_init and not isinstance(ResNetLayer, self):
            assert w_init.shape == (in_shape + 1, out_shape)
            self.W: Tensor = w_init if w_init else Tensor(np.random.randn(in_shape + 1, out_shape) * 0.01)
        self.Z: np.ndarray = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass of the layer
        Args:
            X (np.ndarray): input of the layer
        """
        m = X.shape[1]
        X = pad(X)
        self.Z = self.activation(self.W.T @ X)
        return self.Z

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

class ResNetLayer(Layer):
    def __init__(self, in_shape, out_shape, activation: Activation, w_init = None):
        super().__init__(in_shape, out_shape, activation, w_init)
        if w_init:
            assert w_init.shape == (in_shape + 1, in_shape, 2)
        self.W: Tensor = w_init if w_init else Tensor(np.random.randn(in_shape + 1, out_shape, 2) * 0.01)
        self.inner_Z: np.ndarray = None

    def forward(self, X: np.ndarray):
        W1 = self.W[:, :, 0]
        W2 = self.W[:, :, 1]
        out = self.activation(W1.T @ pad(X))
        self.inner_Z = out
        out = self.activation(W2.T @ pad(X + out))
        self.Z = out
        return out
        

if __name__ == "__main__":
    pass
    #NN
    # Layer1 = Layer(np.random.rand(3, 3), ReLU())
    # Layer2 = Layer(np.random.rand(3, 3), ReLU())
    # Layer3 = Layer(np.random.rand(3, 3), SoftMax())
    
    # forward1 = Layer1(np.random.rand(3, 3))
    # forward2 = Layer2(forward1)
    # forward3 = Layer3(forward2)
    # print(forward3)
    
    # Layer3.backward(forward3, prev_grad= np.array([1, 1, 1]))
    # grad_accumlation = Layer3.grad_x
    # Layer2.backward(forward2, prev_grad=grad_accumlation)
    # grad_accumlation = np.dot(Layer2.grad_x.T, grad_accumlation)
    # Layer1.backward(forward1, prev_grad=grad_accumlation)

    # def backward(self, X : np.ndarray, V : np.ndarray, C : np.ndarray= None):
    #     """Backward pass of the layer
    #     Args:
    #         X (np.ndarray): input of the layer
    #         V (np.ndarray): accumulated gradient from the next layer
    #         C (np.ndarray): target values for the last layer
    #     """
    #     print(f"{X.shape=}, {V.shape=}, {self.W.shape=}, {self.Z.shape=}")
    #     if C is not None:
    #         grad_x = self.activation.grad_X(X, self.W, self.Z, C) # (n, m)
    #         grad_w = self.activation.grad_W(X, self.W, self.Z, C) # (n, k)
    #     else:
    #         activation_grad_x = self.activation.grad_X(X, self.W, self.Z) # (n, m)
    #         activation_grad_w = self.activation.grad_W(X, self.W, self.Z) # (n, k)
    #         print(f"{activation_grad_x.shape=}")
    #         print(f"{activation_grad_w.shape=}")

    #         grad_mul_V_x = activation_grad_x * V # (n, m)
    #         print(f"{grad_mul_V_x.shape=}")

    #         grad_x = np.dot(grad_mul_V_x, self.W.T) 
    #         print(f"{grad_x.shape=}")

    #         grad_mul_V_w = activation_grad_w * V
    #         print(f"{grad_mul_V_w.shape=}")

    #         grad_w = np.dot(X.T, grad_mul_V_w)
    #         print(f"{grad_w.shape=}")
    #     # else:
    #     #     print(f"{C.shape=}")
    #     #     m = self.Z.shape[0]
    #     #     loss = (1 / m) * (self.Z - C)
    #     #     grad_x = np.dot(loss, self.W.T)
    #     #     print(f"{grad_x.shape=}")
    #     #     grad_w = np.dot(X.T, loss)
    #     #     print(f"{grad_w.shape=}")

    #     self._grad_x = grad_x # accumlated grad_x
    #     self._grad_w = grad_w # accumlated grad_w with later grad_x