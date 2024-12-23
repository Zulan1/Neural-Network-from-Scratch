import numpy as np

class Tensor(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.grad = None
        obj.v = np.zeros_like(obj)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grad = getattr(obj, 'grad', None)
        self.v = getattr(obj, 'v', None)

