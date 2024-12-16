import numpy as np

from tqdm import tqdm

from nn import NeuralNetwork
from layer import Layer, ResNetLayer
from loss import Loss, CrossEntropy
from optimizer import Optimizer, SGD
from activations import ReLU, SoftMax

def train_epoch(model: NeuralNetwork, optimizer: Optimizer, loss_fn: Loss, 
                X: np.ndarray, C: np.ndarray, batch_size: int):
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
    m = X.shape[1]
    num_batches = max(m // batch_size, 1)
    Xs = np.array_split(X, num_batches, axis=1)
    Cs = np.array_split(C, num_batches, axis=1)
    for X_batch, C_batch in zip(Xs, Cs):
        # Forward pass
        out = model(X_batch)

        # Backward pass
        loss = loss_fn(out, C_batch)
        loss_fn.backward(model, X_batch, C_batch)


        # Update weights
        optimizer.step(model)
        running_avg_loss += loss / m

    return running_avg_loss

def train(model: NeuralNetwork, epochs: int, batch_size: int,
          loss_fn: Loss = CrossEntropy(), optimizer: Optimizer = SGD(1e-3),
          X: np.ndarray = None, C: np.ndarray = None):
    """
    Train the model using the given optimizer.

    Parameters:
        model (NeuralNetwork): The model to train.
        epochs (int): The number of epochs to train the model for.
        lr (float): The learning rate for the optimizer.
        optimizer (Optimizer): The optimizer to use for training.
    """
    losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        # Forward pass
        loss = train_epoch(model, optimizer, loss_fn, X, C, batch_size)
        losses.append(loss)

    return losses

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