import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from tqdm import tqdm

from nn.loss import CrossEntropy
from nn.optimizer import SGD_momentum
from nn import NeuralNetwork
from train import train_epoch
from metrics import accuracy

if __name__ == "__main__":
    data = loadmat('Data/PeaksData.mat')
    X_train, X_test = data['Yv'], data['Yt']
    C_train, C_test = data['Cv'], data['Ct']
    in_shape = X_train.shape[0]
    out_shape = C_train.shape[0]
    batch_size = 64
    epochs = 100
    model = NeuralNetwork()
    optimizer = SGD_momentum(1e-3, 0.1)
    f: CrossEntropy = CrossEntropy()
    model.add_layer(in_shape, out_shape, 'softmax')
    train_accs, val_accs = [], []

    for _ in tqdm(range(epochs), desc="Training"):
        loss = train_epoch(model, optimizer, f, X_train, C_train, batch_size)
        acc = accuracy(model(X_train), C_train)
        val_acc = accuracy(model(X_test), C_test)
        train_accs.append(acc)
        val_accs.append(val_acc)

    plt.plot(range(epochs), train_accs, label='Train')
    plt.plot(range(epochs), val_accs, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
