import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from tqdm import tqdm

from nn.loss import CrossEntropy
from nn.optimizer import SGD
from nn import NeuralNetwork
from train import train_epoch
from metrics import accuracy


if __name__ == "__main__":
    data = loadmat('Data/SwissRollData.mat')
    X_train, X_test = data['Yv'], data['Yt']
    C_train, C_test = data['Cv'], data['Ct']
    in_shape = X_train.shape[0]
    out_shape = C_train.shape[0]
    os.makedirs('./figures', exist_ok=True)

    lr, batch_size = 1e-2, 32

    epochs = 1000
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
    plt.title(f'Softmax Regression {lr=}, {batch_size=}')

    plt.savefig('./figures/softmax_regression.png')
    plt.show()

    
    
