from typing import List
import numpy as np
from layer import Layer, ResNetLayer
from activations import Tanh, ReLU, SoftMax

class NeuralNetwork():
    """
    A simple neural network implementation from scratch.
    Attributes:
    -----------
    layers : list
        A list to store the layers of the neural network.
    Methods:
    --------
    __init__():
        Initializes the neural network with an empty list of layers.
    forward(X):
        Performs a forward pass through the network with input data X.
    __call__(X):
        Allows the instance to be called as a function to perform a forward pass.
    add_layer(layer):
        Adds a new layer to the neural network.
    """

    def __init__(self):
        self.layers : List[Layer] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X
        for layer in self.layers:
            out = layer(out)
        return out

    def __call__(self, X):
        return self.forward(X)

    def add_layer(
        self, 
        in_shape: int, 
        out_shape: int, 
        activation_type: str, 
        type: str = 'Linear'
    ) -> None:
        assert activation_type.lower() in ['tanh', 'relu', 'softmax'], "Activation function not supported"
        assert not self.layers or self.layers[-1].W.shape[1] == in_shape, \
        f"Input shape mismatch at layer {len(self.layers) + 1}," \
        f"expected {self.layers[-1].W.shape[1]} but got {in_shape}"

        match activation_type.lower():
            case 'tanh':
                activation = Tanh()
            case 'relu':
                activation = ReLU()
            case 'softmax':
                activation = SoftMax()
            case _:
                raise ValueError("Activation function not supported")
        if type == 'Linear':
            layer = Layer(in_shape, out_shape, activation)
        else:
            layer = ResNetLayer(in_shape, out_shape, activation)
        self.layers.append(layer)



if __name__ == "__main__":
    pass