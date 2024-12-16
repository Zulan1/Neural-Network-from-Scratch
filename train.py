import numpy as np
from nn import NeuralNetwork
from layer import Layer, ResNetLayer
from loss import Loss, CrossEntropy
from optimizer import Optimizer, SGD, SGD_momentum
from activations import Activation, ReLU, Tanh, SoftMax

def train_epoch(model: NeuralNetwork, lr: float, optimizer: Optimizer, loss_fn: Loss, Xs: np.ndarray, Cs: np.ndarray):
    """
    Train the model for one epoch.

    Parameters:
        model (NeuralNetwork): The model to train.
        lr (float): The learning rate for the optimizer.
        optimizer (Optimizer): The optimizer to use for training.
        loss_fn (Loss): The loss function to use for training.
        X (np.ndarray): The input data.
        C (np.ndarray): The target data.
    """
    running_avg_loss = 0
    for X, C in zip(Xs, Cs):
        # Forward pass
        out = model(X)
        loss = loss_fn(out, C)
        # Backward pass
        loss.backward(model, C, X)

        # Update weights
        optimizer.step(model)
        running_avg_loss += loss / (len(Xs) * Xs.shape[2]) # M/m * m

    return running_avg_loss

def train(model: NeuralNetwork, epochs: int, lr: float = 1e-3,
          loss_fn: Loss = None,
          Xs: np.ndarray = None, Cs: np.ndarray = None):
    """
    Train the model using the given optimizer.

    Parameters:
        model (NeuralNetwork): The model to train.
        epochs (int): The number of epochs to train the model for.
        lr (float): The learning rate for the optimizer.
        optimizer (Optimizer): The optimizer to use for training.
    """
    optimizer = SGD(lr)
    for epoch in range(epochs):
        # Forward pass
        loss = train_epoch(model, lr, optimizer, loss_fn, Xs, Cs)
        print(f"Epoch {epoch} - Loss: {loss}")


if __name__ == "__main__":
    L = 3
    lr = 1e-3
    epochs = 100


    X_train, X_test, X_val = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    X_train.split () # -> (M/m, n ,m)
    C = np.array([[0, 1], [1, 0], [1, 0], [0, 1]]).T

    model = NeuralNetwork()
    for _ in range(L - 1):
        model.add_layer(Layer(2, 2, ReLU()))
    model.add_layer(Layer(2, 1, SoftMax()))
    loss_fn = CrossEntropy()
    
    train(model, epochs, lr, loss_fn, X_train, C)