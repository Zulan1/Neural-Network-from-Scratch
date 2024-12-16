import numpy as np
from layer import Layer
from activations import Activation, ReLU, SoftMax, Tanh
from SGD import SGD, SGD_momentum

class NeuralNetwork(): #todo Dvir

    def __init__(self, solver):
        self.layers = []
        self.solver = solver
        self.X_layers = []

        # for i in range(L - 1):
        #     self.layers.append(Layer(input_shape, input_shape // 2, activations[i], solver = solver))
        #     input_shape = input_shape // 2
        #     if input_shape <= num_classes:
        #         break

        # assert isinstance(activations[-1], SoftMax)
        # num_classes = num_classes if num_classes != 2 else 1
        # self.layers.append(Layer(input_shape, num_classes, activations[-1], solver = solver))

    def forward(self, X):
        self.X_layers = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.X_layers.append(X)
        return X

    def __call__(self, X):
        self.forward(X)
    
    def add_layer(self, layer):
        self.layers.append(layer)

if __name__ == "__main__":
    pass