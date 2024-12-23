import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lr: float):
        self.lr = lr

    def step(self, model) -> None:
        for layer in model.layers:
            layer.W = self.update(layer.W)

    @abstractmethod
    def update(self, W: np.ndarray) -> np.ndarray:
        pass

class SGD(Optimizer):
    def update(self, W):
        assert W.grad is not None, "Didn't find a gradient, did you forget to call loss.backward()?"       
        return W - self.lr * W.grad

class SGD_momentum(Optimizer):
    def __init__(self, lr: float, beta: float):
        super().__init__(lr)
        self.beta = beta

    def update(self, W):
        assert W.grad is not None, "Didn't find a gradient, did you forget to call loss.backward()?"
        W.v = self.beta * W.v - self.lr * W.grad
        return W + W.v


if __name__ == "__main__":
    opt = SGD(0.1)
    m = 10
    n = 1
    X = np.random.rand(m, n)
    y = 5 * X + 3
    W = np.random.rand(m, 2)
    X = np.vstack([X, np.ones(n)])


