import numpy as np
import click
import matplotlib.pyplot as plt

from scipy.io import loadmat
from typing import List

from nn import NeuralNetwork
from nn.loss import CrossEntropy
from nn.optimizer import SGD, SGD_momentum
from plots import plot_train_losses, plot_model_decision_boundries
from train import train
from metrics import accuracy


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
@click.option('--net_shape', default='2,2x15,2', callback=parse_list, help='Number of layers in the model (comma-separated integers).')
@click.option('--activation', default='relu', type=click.Choice(['tanh', 'relu']), help='Activation function to use.')
@click.option('--resnet', is_flag=True, default=False, type=bool, help='True of using ResNet layers')
@click.option('--loss', default='crossentropy', type=click.Choice(['crossentropy']), help='Loss function to use.')
@click.option('--optim', default='sgd', type=click.Choice(['sgd', 'sgd_momentum']), help='Optimizer to use.')
@click.option('--batch_size', default=512, type=int, help='Batch size for training.')
@click.option('--epochs', default=100, type=int, help='Number of epochs to train the model for.')
@click.option('--lr', default=1e-2, type=float, help='Learning rate for the optimizer.')
@click.option('--beta', default=0.1, type=float, help='Momentum for the optimizer.')


def main(data_path: str, 
         net_shape: List[int], 
         activation: str,
         resnet: bool,
         loss: str, 
         optim: str, 
         batch_size: int, 
         epochs: int, 
         lr: float, 
         beta: float):
    # Load the data
    data = loadmat(data_path)
    X_train, C_train = data['Yt'], data['Ct']
    X_test, C_test = data['Yv'], data['Cv']

    print(f"{X_train.shape=}, {C_train.shape=}")
    print(f"{X_test.shape=}, {C_test.shape=}")
    print(f"{net_shape=}")

    # Initialize the model
    model = NeuralNetwork()
    input_dim = X_train.shape[0]
    output_dim = C_train.shape[0]
    assert net_shape[0] == input_dim, \
    f"First layer must have the same shape as the input data. Expected {input_dim}, got {net_shape[0]}"
    assert net_shape[-1] == output_dim, \
        f"Last layer must have the same shape as the output data. Expected {output_dim}, got {net_shape[-1]}"
    if resnet:
        assert len(set(net_shape[1:-1])) == 1, "ResNet layers must have the same number of units in each hidden layer"
    L = len(net_shape) - 1
    for i in range(L):
        input_dim, output_dim = net_shape[i], net_shape[i + 1]
        if i == L - 1:
            model.add_layer(input_dim, output_dim, 'softmax')
        elif i != 0:
            model.add_layer(input_dim, output_dim, activation, resnet)
        else:
            model.add_layer(input_dim, output_dim, activation)

    # Select the optimizer
    if optim == 'sgd':
        optimizer_fn = SGD(lr)
    elif optim == 'sgd_momentum':
        if beta is None:
            raise ValueError("Momentum requires a beta value.")
        optimizer_fn = SGD_momentum(lr, beta)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    # Select the loss function
    if loss == 'crossentropy':
        loss_fn = CrossEntropy()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")


    plot_train_losses(model=model, loss_fn=loss_fn,
                       optimizer_fn=optimizer_fn, epochs=epochs,
                         batch_size=batch_size, X_train=X_train,
                           C_train=C_train, X_val=X_test, C_val=C_test)
    plot_model_decision_boundries(model, X_test, C_test)
    # Train the model
    # losses, val_losses = train(model,\
    #                            epochs,
    #                            batch_size,
    #                            loss_fn,
    #                            optimizer_fn,
    #                            X_train,
    #                            C_train,
    #                            X_test,
    #                            C_test,
    #                            )
    # probs = model(X_test)
    # acc = accuracy(C_test, probs)
    # _, ax = plt.subplots(1, 2)

    # minx, miny = np.min(X_train, axis=1)
    # maxx, maxy = np.max(X_train, axis=1)
    # xx, yy = np.meshgrid(np.linspace(minx, maxx, 100), np.linspace(miny, maxy, 100))
    # X_grid = np.vstack([xx.ravel(), yy.ravel()])
    # C_grid = model(X_grid)
    # C_grid = np.argmax(C_grid, axis=0)
    # print(f"Loss: {losses[-1]}")
    # print(f"Accuracy: {acc}")
    # ax[0].contourf(xx, yy, C_grid.reshape(xx.shape), levels=np.arange(C_train.shape[0] + 1) - 0.5, alpha=0.5)
    # ax[1].scatter(*X_test, c=np.argmax(C_test, axis=0), cmap='viridis')
    # plt.show()
    # plt.plot(range(epochs), losses, label='Training Loss', color='k')
    # if val_losses:
    #     plt.plot(range(epochs), val_losses, label='Validation Loss', color='r')
    # plt.legend()
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.show()
    


if __name__ == "__main__":
    main()