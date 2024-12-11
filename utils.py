import numpy as np

def GradientTest(x, f, grad):
    dx = np.random.rand(1e-5, 1e-4)
    left = f(x + dx) # + f(x) ?
    right = grad * dx
    print(f"f(x + dx)={left}")
    print(f"grad * dx={right}")
    return np.allclose(left, right)

def JacobianTest(x, f, jacob):
    dx = np.random.rand(1e-5, 1e-4)
    left = f(x + dx) # + f(x) ?
    right = jacob * dx
    print(f"f(x + dx)={left}")
    print(f"jacob^T * dx={right}")

if __name__ == "__main__": #Dvir
    pass