import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from nn import NeuralNetwork
from nn.loss import Loss
from nn.optimizer import Optimizer
from train import train_epoch
from metrics import accuracy, f1_score, precision, recall


def plot_train_losses(model: NeuralNetwork, loss_fn: Loss, optim: Optimizer,
             epochs: int, batch_size: int,
             X_train: np.ndarray, C_train: np.ndarray,
             X_val: np.ndarray, C_val: np.ndarray, save_path: str):
    """
    Train a neural network model and display an interactive graph of training
    and validation losses.

    Args:
        model (NeuralNetwork): The neural network model to train.
        loss_fn (Loss): The loss function used for training.
        optim (Optimizer): The optimizer used for parameter updates.
        epochs (int): Number of epochs to train the model.
        batch_size (int): Batch size for training.
        X_train (np.ndarray): Training data features.
        C_train (np.ndarray): Training data labels.
        X_val (np.ndarray): Validation data features.
        C_val (np.ndarray): Validation data labels.
    """
    # Initialize variables to store losses
    train_losses = []
    val_losses = []

    # Prepare the interactive plot
    _, ax = plt.subplots()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    train_line, = ax.plot([], [], label='Train Loss', color='blue')
    val_line, = ax.plot([], [], label='Validation Loss', color='orange')
    plt.legend()

    # Add a text box for learning rate
    lr_text = ax.text(0.1, 0.95, '', transform=ax.transAxes)
    loss_text = ax.text(0.1, 0.85, '', transform=ax.transAxes)

    # Helper function to update the graph
    def update_graph(train_loss, val_loss, lr, t):
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_line.set_data(range(len(train_losses)), train_losses)
        val_line.set_data(range(len(val_losses)), val_losses)
        ax.relim()
        ax.autoscale_view()
        lr_text.set_text(f'Learning Rate: {lr / np.sqrt(t):.6f}')
        loss_text.set_text(f'Train Loss: {train_loss:.2f}\nVal Loss: {val_loss:.2f}')
        # print(f"{lr=}")
        plt.draw()
        plt.pause(0.01)

    for i in tqdm(range(epochs), desc="Training Model"):
        epoch_train_loss = train_epoch(model, optim, loss_fn, X_train, C_train, batch_size)  # Set model to training mode

        val_predictions = model(X_val)
        epoch_val_loss = loss_fn(val_predictions, C_val)

        update_graph(epoch_train_loss, epoch_val_loss, optim.lr, optim.t)

        if i > 0.9 * epochs and val_losses[-1] > np.min(val_losses):
            print(f"Early stopping at epoch {i}")
            break

    plt.savefig(f'{save_path}_loss_plots.png')
    plt.show()

    _, acc, _, _, _ = evaluate_model(model, X_val, C_val, loss_fn)

    print(f"Training complete.\nModel final loss: {train_losses[-1]:.2f}\nValidation final loss: {val_losses[-1]:.2f}\nTraining accuracy: {acc:.2f}")


def plot_model_decision_boundries(model: NeuralNetwork, X: np.ndarray, C: np.ndarray, save_path: str):
    """
    Plot the decision boundaries of a neural network model.

    Args:
        model (NeuralNetwork): The trained neural network model.
        X (np.ndarray): Input data features.
        C (np.ndarray): Input data labels.
    """
    min_x, min_y = np.min(X, axis=1)
    max_x, max_y = np.max(X, axis=1)
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, 100), np.linspace(min_y, max_y, 100))
    X_grid = np.vstack([xx.ravel(), yy.ravel()])
    C_grid = model(X_grid)
    C_grid = np.argmax(C_grid, axis=0)

    _, ax = plt.subplots(1, 3)

    ax[0].contourf(xx, yy, C_grid.reshape(xx.shape), levels=np.arange(C.shape[0] + 1) - 0.5, alpha=0.5)
    ax[1].scatter(*X, c=np.argmax(C, axis=0), cmap='viridis')
    ax[2].contourf(xx, yy, C_grid.reshape(xx.shape), levels=np.arange(C.shape[0] + 1) - 0.5, alpha=0.5)
    ax[2].scatter(*X, c=np.argmax(C, axis=0), cmap='viridis')
    plt.savefig(f'{save_path}_predictions.png')
    plt.show()
    
def evaluate_model(model: NeuralNetwork, X: np.ndarray, C: np.ndarray, loss_fn: Loss):
    """
    Evaluate the model on the given data.

    Args:
        model (NeuralNetwork): The trained neural network model.
        X (np.ndarray): Input data features.
        C (np.ndarray): Input data labels.
    """
    predictions = model(X)
    loss = loss_fn(predictions, C)
    acc = accuracy(predictions, C)
    f1 = f1_score(predictions, C)
    prec = precision(predictions, C)
    rec = recall(predictions, C)
    print(f"Model loss: {loss:.2f}\nAccuracy: {acc:.2f}\nF1 Score: {f1:.2f}\nPrecision: {prec:.2f}\nRecall: {rec:.2f}")
    return loss, acc, f1, prec, rec