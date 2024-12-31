import numpy as np

from tqdm import tqdm

from nn import NeuralNetwork
from nn.loss import Loss, CrossEntropy
from nn.optimizer import Optimizer, SGD


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
    from utils import split_into_batches

    running_avg_loss = 0
    m = X.shape[1]
    
    indices = np.arange(m)
    np.random.shuffle(indices)
    
    X, C = X[:, indices], C[:, indices]
    Xs = split_into_batches(X, batch_size)
    Cs = split_into_batches(C, batch_size)

    optimizer.t += 1

    for X_batch, C_batch in zip(Xs, Cs):
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
    for i in tqdm(range(epochs), desc="Training"):
        # Forward pass
        loss = train_epoch(model, optimizer, loss_fn, X, C, batch_size)
        val_loss = loss_fn(model(X_val), C_val) if X_val is not None else None
        losses.append(loss)
        if val_loss is None:
            continue

        val_losses.append(val_loss)

        if i > 0.98 * epochs and val_losses[-1] < np.min(val_losses):
            print(f"Early stopping at epoch {i}")
            break

    return losses, val_losses

if __name__ == "__main__":
    pass