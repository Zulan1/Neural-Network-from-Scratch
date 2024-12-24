import numpy as np
import click
import matplotlib.pyplot as plt

from scipy.io import loadmat
from typing import List

from utils import nn_builder
from utils import plot_train_losses, plot_model_decision_boundries
from utils import train_test_split


def parse_list(_, __, value):
    """Callback to parse a comma-separated string into a list of integers."""
    try:
        lis = []
        for i in value.split(','):
            if 'x' in i:
                i = i.split('x')
                for _ in range(int(i[0])):
                    lis.append(int(i[1]))
            else:
                lis.append(int(i))
        return lis
    except ValueError:
        raise click.BadParameter('List must contain integers separated by commas.')

@click.command()
@click.option('--data_path', default='data/SwissRollData.mat', type=str, help='Path to the data directory.')
@click.option('--net_shape', default='2x15', callback=parse_list, help='Width of intermediate layers (comma-separated integers).')
@click.option('--activation', default='relu', type=click.Choice(['tanh', 'relu']), help='Activation function to use.')
@click.option('--resnet', is_flag=True, default=False, type=bool, help='True of using ResNet layers')
@click.option('--loss', default='crossentropy', type=click.Choice(['crossentropy']), help='Loss function to use.')
@click.option('--optim', default='sgd', type=click.Choice(['sgd']), help='Optimizer to use.')
@click.option('--batch_size', default=512, type=int, help='Batch size for training.')
@click.option('--epochs', default=100, type=int, help='Number of epochs to train the model for.')
@click.option('--lr', default=1e-2, type=float, help='Learning rate for the optimizer.')
@click.option('--momentum', default=0.0, type=float, help='Momentum for the optimizer.')


def main(data_path: str, 
         net_shape: List[int], 
         activation: str,
         resnet: bool,
         loss: str, 
         optim: str,
         batch_size: int, 
         epochs: int, 
         lr: float, 
         momentum: float):
    # Load the data
    data = loadmat(data_path)
    X_train, C_train = data['Yt'], data['Ct']
    X_train, C_train, X_val, C_val = train_test_split(X_train, C_train, test_size=0.2)
    X_test, C_test = data['Yv'], data['Cv']

    print(f"{X_train.shape=}, {C_train.shape=}")
    print(f"{X_test.shape=}, {C_test.shape=}")
    print(f"{net_shape=}")

    # Initialize the model
    input_dim = X_train.shape[0]
    output_dim = C_train.shape[0]
    net_shape = [input_dim] + net_shape + [output_dim]
    assert net_shape[0] == input_dim, \
    f"First layer must have the same shape as the input data. Expected {input_dim}, got {net_shape[0]}"
    assert net_shape[-1] == output_dim, \
        f"Last layer must have the same shape as the output data. Expected {output_dim}, got {net_shape[-1]}"
    if resnet:
        assert len(set(net_shape[1:-1])) == 1, "ResNet layers must have the same number of units in each hidden layer"

    model, optimizer_fn, loss_fn = nn_builder(net_shape, activation, resnet, loss, optim, lr, momentum)

    train_args = {
        'model': model,
        'loss_fn': loss_fn,
        'optim': optimizer_fn,
        'epochs': epochs,
        'batch_size': batch_size,
        'X_train': X_train,
        'C_train': C_train,
        'X_val': X_val,
        'C_val': C_val
    }

    plot_train_losses(**train_args)
    plot_model_decision_boundries(model, X_test, C_test)    


if __name__ == "__main__":
    main()