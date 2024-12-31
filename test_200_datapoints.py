import numpy as np

from scipy.io import loadmat
from tqdm import tqdm
from nn.loss import CrossEntropy
from nn.optimizer import SGD
from nn import NeuralNetwork
from utils import nn_builder, plot_train_losses, plot_model_decision_boundries, train_test_split, evaluate_model


if __name__ == "__main__":
    dataset = 'Peaks'
    data = loadmat(f"Data/{dataset}Data.mat")
    save_path = f'figures/200_datapoints_{dataset}'
    X_train, X_test = data['Yv'], data['Yt']
    C_train, C_test = data['Cv'], data['Ct']
    X_train, C_train, X_val, C_val = train_test_split(X_train, C_train, test_size=0.2)

    indices = np.random.choice(np.arange(X_train.shape[1]), 200, replace=False)
    X_train_200 = X_train[:,indices]
    C_train_200 = C_train[:,indices]

    print(f"{X_train_200.shape=}, {C_train_200.shape=}")
    print(f"{np.array([C_train_200[0] == 1.]).sum()=}")

    model, optimizer_fn, loss_fn = nn_builder([2,15,15,5], "relu", False, "crossentropy", "sgd", 0.1, 0.9)

    train_args = {
        'model': model,
        'loss_fn': loss_fn,
        'optim': optimizer_fn,
        'epochs': 1000,
        'batch_size': 20,
        'X_train': X_train,
        'C_train': C_train,
        'X_val': X_val,
        'C_val': C_val,
        'save_path': save_path
    }

    plot_train_losses(**train_args)
    if dataset != 'GMM':
        plot_model_decision_boundries(model, X_test, C_test, save_path)
    evaluate_model(model, X_test, C_test, loss_fn)

