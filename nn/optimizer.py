import numpy as np

from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lr: float, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.t = 0

    def step(self, model) -> None:
        for layer in model.layers:
            layer.W = self.update(layer.W)

    @abstractmethod
    def update(self, W: np.ndarray) -> np.ndarray:
        pass

class SGD(Optimizer):
    def update(self, W):
        assert W.grad is not None, "Didn't find a gradient, did you forget to call loss.backward()?"
        eta = self.lr / np.sqrt(self.t)
        W.v = self.momentum * W.v - eta * W.grad
        return W + W.v


if __name__ == "__main__":
    pass


