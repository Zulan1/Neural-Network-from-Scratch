import matplotlib.pyplot as plt
import optuna
import os

from scipy.io import loadmat
from tqdm import tqdm

from nn.loss import CrossEntropy
from nn.optimizer import SGD
from nn import NeuralNetwork
from train import train_epoch
from metrics import accuracy

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [2 ** i for i in range(10)])
    lr = trial.suggest_float('lr', 1e-5, 1)

    print(f"Starting trial {trial.number} with params {trial.params}")
    epochs = 100
    model = NeuralNetwork()
    model.add_layer(in_shape, out_shape, 'softmax')
    optimizer = SGD(lr, momentum=0.9)
    loss_fn: CrossEntropy = CrossEntropy()
    train_accs, val_accs = [], []

    for _ in tqdm(range(epochs), desc="Training"):
        train_epoch(model, optimizer, loss_fn, X_train, C_train, batch_size)
        acc = accuracy(model(X_train), C_train)
        val_acc = accuracy(model(X_test), C_test)
        train_accs.append(acc)
        val_accs.append(val_acc)
    
    return val_accs[-1]



if __name__ == "__main__":
    data = loadmat('Data/SwissRollData.mat')
    X_train, X_test = data['Yv'], data['Yt']
    C_train, C_test = data['Cv'], data['Ct']
    in_shape = X_train.shape[0]
    out_shape = C_train.shape[0]
    os.makedirs('./figures', exist_ok=True)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    print(f"{study.best_params=}")

    batch_size = study.best_params['batch_size']
    lr = study.best_params['lr']

    epochs = 100
    model = NeuralNetwork()
    model.add_layer(in_shape, out_shape, 'softmax')
    optimizer = SGD(lr, momentum=0.9)
    loss_fn: CrossEntropy = CrossEntropy()
    train_accs, val_accs = [], []

    for _ in tqdm(range(epochs), desc="Training"):
        train_epoch(model, optimizer, loss_fn, X_train, C_train, batch_size)
        acc = accuracy(model(X_train), C_train)
        val_acc = accuracy(model(X_test), C_test)
        train_accs.append(acc)
        val_accs.append(val_acc)

    plt.plot(range(epochs), train_accs, label='Train')
    plt.plot(range(epochs), val_accs, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig('./figures/softmax_regression.png')
    plt.show()

    
    
