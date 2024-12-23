import optuna
import numpy as np

from scipy.io import loadmat
from tqdm import tqdm
from nn.loss import CrossEntropy
from nn.optimizer import SGD
from nn import NeuralNetwork
from train import train_epoch
from utils import nn_builder


def optuna_train(X: np.ndarray, C: np.ndarray, batch_size: int, epochs: int):
    def objective(trial):
        in_shape = X.shape[0]
        out_shape = C.shape[0]
        L = trial.suggest_int('L', 1, 5)
        resnet = trial.suggest_categorical('resnet', [True, False])
        net_shape = [in_shape]
        if resnet:
            width = trial.suggest_int('resnet_width', 1, 50)
            net_shape += [width] * L
        else:
            for i in range(L):
                l_shape = trial.suggest_int(f'layer_{i}', 1, 50)
                net_shape.append(l_shape)
        net_shape.append(out_shape)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        momentum = trial.suggest_categorical('momentum', list(np.arange(0, 1, 0.1)))
        lr = trial.suggest_categorical('lr', [10 ** i for i in range(-5, 0)] + [0.5 * 10 ** i for i in range(-5, 0)])

        model, optim_fn, loss_fn = nn_builder(net_shape, activation, resnet, 'crossentropy', 'sgd', lr, momentum)

        tqdm.write(f"Starting trial {trial.number} with params {trial.params}")

        k = 5
        indices = np.arange(X.shape[1])
        np.random.shuffle(indices)
        fold_indices = np.array_split(indices, k)
        val_losses = []

        for i in tqdm(range(k), desc="Cross-validation"):
            val_indices = fold_indices[i]
            train_indices = np.concatenate([fold_indices[j] for j in range(k) if j != i])
            X_train, C_train = X[:, train_indices], C[:, train_indices]
            X_val, C_val = X[:, val_indices], C[:, val_indices]

            for _ in tqdm(range(epochs), desc="Training", leave=False):
                train_epoch(model, optim_fn, loss_fn, X_train, C_train, batch_size)

            val_loss = loss_fn(model(X_val), C_val)
            val_losses.append(val_loss)
            

        return np.mean(val_losses)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)
    return study.best_params


if __name__ == "__main__":
    data = loadmat('Data/PeaksData.mat')
    X_train, X_test = data['Yv'], data['Yt']
    C_train, C_test = data['Cv'], data['Ct']
    batch_size = 512
    epochs = 2000
    best_params = optuna_train(X_train, C_train, batch_size, epochs)
    print(f"{best_params=}")
