from typing import List
import numpy as np
from layer import Layer

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

    def forward(self, X : np.ndarray):
        out = X
        for layer in self.layers:
            out = np.concatenate((out, np.ones((1, out.shape[1]))), axis=0)
            out = layer(out)
        return out

    def __call__(self, X):
        self.forward(X)

    def add_layer(self, layer):
        self.layers.append(layer)



if __name__ == "__main__":
    pass