import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj._grad = None
        obj._prev_grad = np.zeros_like(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._grad = getattr(obj, '_grad', None)
        self._prev_grad = getattr(obj, '_prev_grad', None)

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        if self._grad is not None:
            self._prev_grad = self._grad
        self._grad = value

    @property
    def prev_grad(self):
        return self._prev_grad

    @prev_grad.setter
    def prev_grad(self, _):
        pass

