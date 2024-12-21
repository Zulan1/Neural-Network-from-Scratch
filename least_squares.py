import numpy as np
import matplotlib.pyplot as plt

from layer import Layer
from tensor import Tensor
from optimizer import SGD_momentum
from utils import pad

if __name__ == "__main__":
    n = 1
    m = 10
    X = np.random.rand(n, m)
    Y = 5 * X + 3
    W = Tensor(np.random.randn(n + 1, 1))
    lr = 1e-2
    beta = 0.1
    losses = []
    optim = SGD_momentum(lr, beta)
    layer = Layer(n, 1, lambda x: x)


    for i in range(500):
        W = layer.W.view()
        layer.W.grad = -2 * pad(X) @ (Y - W.T @ pad(X)).T
        optim.step(layer)
        loss = np.linalg.norm(Y - W.T @ pad(X)) ** 2
        losses.append(loss)
    
    plt.scatter(X.ravel(), Y.ravel(), marker='x', label='True')
    plt.plot(X.ravel(), (W.T @ pad(X)).ravel(), label='Predicted')
    plt.legend()
    plt.show()

