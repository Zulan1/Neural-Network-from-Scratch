import numpy as np

from tqdm import tqdm

from nn import NeuralNetwork
from nn.layer import Layer
from nn.loss import Loss, CrossEntropy
from nn.optimizer import Optimizer, SGD
from nn.activations import ReLU, SoftMax


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
    num_batches = ((m - 1) // batch_size) + 1
    Xs = np.array_split(X, num_batches, axis=1)
    Cs = np.array_split(C, num_batches, axis=1)
    for X_batch, C_batch in zip(Xs, Cs):
        if X_batch.size == 0:
            continue

        # Forward pass
        out = model(X_batch)

        # Backward pass
        loss = loss_fn(out, C_batch)
        loss_fn.backward(model, X_batch, C_batch)


        # Update weights
        optimizer.step(model)
        running_avg_loss += loss * X_batch.shape[1]

    return running_avg_loss / m

def train(model: NeuralNetwork, epochs: int, batch_size: int,
          loss_fn: Loss = CrossEntropy(), optimizer: Optimizer = SGD(1e-3),
          X: np.ndarray = None, C: np.ndarray = None, 
          X_val: np.ndarray = None, C_val: np.ndarray = None):
    """
    Train the model using the given optimizer.

    Parameters:
        model (NeuralNetwork): The model to train.
        epochs (int): The number of epochs to train the model for.
        lr (float): The learning rate for the optimizer.
        optimizer (Optimizer): The optimizer to use for training.
    """
    losses = []
    val_losses = []
    for _ in tqdm(range(epochs), desc="Training"):
        # Forward pass
        loss = train_epoch(model, optimizer, loss_fn, X, C, batch_size)
        val_loss = loss_fn(model(X_val), C_val) if X_val is not None else None
        losses.append(loss)
        if val_loss is not None:
            val_losses.append(val_loss)

    return losses, val_losses

if __name__ == "__main__":
    pass