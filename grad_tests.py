import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from loss import CrossEntropy
from layer import Layer
from activations import Tanh, ReLU, SoftMax
from tensor import Tensor

def gradient_test(X, C):
    in_shape = X.shape[0]
    out_shape = C.shape[0]
    f: CrossEntropy = CrossEntropy()
    d = np.random.randn(in_shape + 1, out_shape)
    d /= np.linalg.norm(d)
    print(f"{np.linalg.norm(d)=}, {d=}")
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    layer = Layer(in_shape, out_shape, SoftMax())
    W: Tensor = layer.W.view()
    print(f"{W.shape=}, {X.shape=}, {C.shape=}")
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        W += eps * d
        Z = layer(X)
        leff_err = f(Z, C)

        # Compute right term
        W -= eps * d
        Z = layer(X)
        right_err = f(Z, C)

        err = leff_err - right_err
        no_grad__abs_err: float = np.abs(err)
        f.backward(layer, X, C)
        dW: np.ndarray = layer.W.grad
        grad_abs_err = np.abs(err - eps * (d.ravel() @ dW.ravel()))

        no_grad_errs.append(no_grad__abs_err)
        grad_errs.append(grad_abs_err)
    
    log_no_grad_errs = np.log(no_grad_errs)
    log_grad_errs = np.log(grad_errs)

    slope_no_grad, _ = np.polyfit(range(n_iter), log_no_grad_errs, 1)
    slope_grad, _ = np.polyfit(range(n_iter), log_grad_errs, 1)


    plt.plot(range(n_iter), no_grad_errs, label='No grad', color='b')
    plt.plot(range(n_iter), grad_errs, label='Grad', color='r')
    plt.text(n_iter - 2, no_grad_errs[-1], f"Slope: {slope_no_grad:.2f}", color='b')
    plt.text(n_iter - 2, grad_errs[-1], f"Slope: {slope_grad:.2f}", color='r')
    plt.yscale('log')
    plt.legend()
    plt.show()

def jacobian_test(X, C):
    in_shape = X.shape[0]
    out_shape = C.shape[0]
    m = X.shape[1]
    f: CrossEntropy = CrossEntropy()
    d = np.random.randn(in_shape + 1, out_shape)
    d /= np.linalg.norm(d)
    print(f"{np.linalg.norm(d)=}, {d=}")
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    layer = Layer(in_shape, out_shape, Tanh())
    W: Tensor = layer.W.view()
    print(f"{W.shape=}, {X.shape=}, {C.shape=}")
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        W += eps * d
        left_Z = layer(X)

        # Compute right term
        W -= eps * d
        right_Z = layer(X)

        err = left_Z - right_Z
        no_grad__abs_err: float = np.linalg.norm(err)
        dZ: np.ndarray = layer.activation.grad(right_Z)
        X_temp = np.concatenate((X, np.ones((1, m))), axis=0)
        print(f"{dZ.shape=}, {d.shape=}, {err=}")
        print(f"{eps=}, {d.shape=}")
        print(f"{err - eps * (d.T @ X_temp * dZ)=}")
        grad_abs_err = np.linalg.norm(err - eps * (d.T @ X_temp * dZ))
        print(f"{grad_abs_err=}, {no_grad__abs_err=}")

        no_grad_errs.append(no_grad__abs_err)
        grad_errs.append(grad_abs_err)
    
    log_no_grad_errs = np.log(no_grad_errs)
    log_grad_errs = np.log(grad_errs)

    slope_no_grad, _ = np.polyfit(range(n_iter), log_no_grad_errs, 1)
    slope_grad, _ = np.polyfit(range(n_iter), log_grad_errs, 1)


    plt.plot(range(n_iter), no_grad_errs, label='No grad', color='b')
    plt.plot(range(n_iter), grad_errs, label='Grad', color='r')
    plt.text(n_iter - 2, no_grad_errs[-1], f"Slope: {slope_no_grad:.2f}", color='b')
    plt.text(n_iter - 2, grad_errs[-1], f"Slope: {slope_grad:.2f}", color='r')
    plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    Data = loadmat('Data/GMMData.mat')
    X, C = Data['Yt'], Data['Ct']
    print(f"{X.shape=}, {C.shape=}")
    jacobian_test(X, C)
    gradient_test(X, C)
