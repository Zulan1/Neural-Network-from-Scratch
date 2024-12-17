import numpy as np
import click
import matplotlib.pyplot as plt

from scipy.io import loadmat
from typing import List

from nn import NeuralNetwork
from loss import CrossEntropy
from optimizer import SGD, SGD_momentum
from utils import train_test_split
from train import train

def parse_list(ctx, param, value):
    """Callback to parse a comma-separated string into a list of integers."""
    try:
        return [int(i) for i in value.split(',')]
    except ValueError:
        raise click.BadParameter('List must contain integers separated by commas.')

@click.command()
@click.option('--data_path', default='data/SwissRollData.mat', type=str, help='Path to the data directory.')
@click.option('--net_shape', default='2,10,10,2', callback=parse_list, help='Number of layers in the model (comma-separated integers).')
@click.option('--activation', default='relu', type=click.Choice(['tanh', 'relu']), help='Activation function to use.')
@click.option('--loss', default='crossentropy', type=click.Choice(['crossentropy']), help='Loss function to use.')
@click.option('--optim', default='sgd', type=click.Choice(['sgd', 'sgd_momentum']), help='Optimizer to use.')
@click.option('--batch_size', default=512, type=int, help='Batch size for training.')
@click.option('--epochs', default=100, type=int, help='Number of epochs to train the model for.')
@click.option('--lr', default=1e-4, type=float, help='Learning rate for the optimizer.')
@click.option('--beta', default=0.1, type=float, help='Momentum for the optimizer.')


def main(data_path: str, 
         net_shape: List[int], 
         activation: str, 
         loss: str, 
         optim: str, 
         batch_size: int, 
         epochs: int, 
         lr: float, 
         beta: float):
    # Load the data
    data = loadmat(data_path)
    X, C = data['Yt'], data['Ct']
    print(f"{X.shape=}, {C.shape=}")
    X_train, X_test, C_train, C_test = train_test_split(X, C, test_size=0.2)
    # X_train, X_val, C_train, C_val = train_test_split(X_train, C_train, test_size=0.2)
    print(f"{C_train=}")

    # Initialize the model
    model = NeuralNetwork()
    input_dim = X_train.shape[0]
    output_dim = C_train.shape[0]
    assert net_shape[0] == input_dim, \
    f"First layer must have the same shape as the input data. Expected {input_dim}, got {net_shape[0]}"
    assert net_shape[-1] == output_dim, \
        f"Last layer must have the same shape as the output data. Expected {output_dim}, got {net_shape[-1]}"
    L = len(net_shape) - 1
    for i in range(L):
        input_dim, output_dim = net_shape[i], net_shape[i + 1]
        if i != L - 1:
            model.add_layer(input_dim, output_dim, activation)
        else:
            model.add_layer(input_dim, output_dim, 'softmax')

    # Select the optimizer
    if optim == 'sgd':
        optimizer_fn = SGD(lr)
    elif optim == 'sgd_momentum':
        optimizer_fn = SGD_momentum(lr, beta)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    # Select the loss function
    if loss == 'crossentropy':
        loss_fn = CrossEntropy()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


    
    # Train the model
    losses = train(model, epochs, batch_size, loss_fn, optimizer_fn, X_train, C_train)
    probs = model(X_test)
    C_test_pred = np.zeros_like(probs)
    C_test_pred[np.argmax(probs, axis=0), np.arange(probs.shape[1])] = 1
    accuracy = np.mean(C_test_pred == C_test)

    minx, miny = np.min(X, axis=1)
    maxx, maxy = np.max(X, axis=1)
    xx, yy = np.meshgrid(np.linspace(minx, maxx, 100), np.linspace(miny, maxy, 100))
    X_grid = np.vstack([xx.ravel(), yy.ravel()])
    C_grid = model(X_grid)
    C_grid = np.argmax(C_grid, axis=0)
    C_grid = np.where(C_grid > 0, 1, -1)
    print(f"Loss: {losses[-1]}")
    print(f"Accuracy: {accuracy}")
    plt.contourf(xx, yy, C_grid.reshape(xx.shape), levels=np.arange(-2, 2), cmap='bwr', alpha=0.5)
    plt.scatter(*X_test, c=np.argmax(C_test, axis=0), cmap='viridis')
    plt.show()
    plt.plot(range(epochs), losses, label='Training Loss', color='k')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    


if __name__ == "__main__":
    main()