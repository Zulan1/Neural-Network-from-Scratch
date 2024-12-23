import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat

from nn.loss import CrossEntropy
from nn.layer import Layer
from nn.activations import Tanh, SoftMax
from nn import NeuralNetwork
from utils import pad
from utils.tensor import Tensor


def gradient_test(X, C):
    in_shape = X.shape[0]
    out_shape = C.shape[0]
    f: CrossEntropy = CrossEntropy()
    d = np.random.randn(in_shape + 1, out_shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    layer = Layer(in_shape, out_shape, SoftMax())
    W: Tensor = layer.W.view()
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        A = layer.activation((W + eps * d).T @ pad(X))
        leff_err = f(A, C)

        # Compute right term
        A = layer.activation(W.T @ pad(X))
        right_err = f(A, C)

        err = leff_err - right_err
        no_grad__abs_err: float = np.abs(err)
        layer(X)
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
    plt.savefig('./plots/grad_test.png')
    plt.show()

def jacobian_test_W(X, C):
    in_shape = X.shape[0]
    out_shape = C.shape[0]
    d = np.random.randn(in_shape + 1, out_shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    layer = Layer(in_shape, out_shape, Tanh())
    W: Tensor = layer.W.view()
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        left_A = layer.activation((W + eps * d).T @ pad(X))
        right_A = layer.activation(W.T @ pad(X))

        err = left_A - right_A
        no_grad__abs_err: float = np.linalg.norm(err)
        dA: np.ndarray = layer.activation.grad(right_A)
        X_temp = pad(X)
        grad_abs_err = np.linalg.norm(err - eps * (d.T @ X_temp * dA))

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

def jacobian_test_X(X, C):
    in_shape = X.shape[0]
    out_shape = C.shape[0]
    d = np.random.randn(*X.shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    layer = Layer(in_shape, out_shape, Tanh())
    W: Tensor = layer.W.view()
    for i in range(1, n_iter + 1):
        eps = 0.5 * (0.5 ** i)

        # Compute left term
        left_A = layer.activation(W.T @ pad(X + eps * d))
        right_A = layer.activation(W.T @ pad(X))

        err = left_A - right_A
        no_grad__abs_err: float = np.linalg.norm(err)
        dA: np.ndarray = layer.activation.grad(right_A)
        grad_abs_err = np.linalg.norm(err - eps * (W.T[:,:-1] @ d * dA))

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

def grad_test_model(X, C):
    model = NeuralNetwork()
    net_shape = [X.shape[0], 10, 10, C.shape[0]]
    loss = CrossEntropy()
    L = len(net_shape) - 1
    for i in range(L):
        input_dim, output_dim = net_shape[i], net_shape[i + 1]
        if i == L - 1:
            model.add_layer(input_dim, output_dim, 'softmax')
        else:
            model.add_layer(input_dim, output_dim, 'tanh')
    W = np.concatenate([layer.W.ravel() for layer in model.layers])
    d = np.random.randn(*W.shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []

    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)
        prev_size = 0
        for i, layer in enumerate(model.layers):
            size = layer.W.size
            layer.W += eps * d[prev_size:prev_size + size].reshape(layer.W.shape)
            prev_size += size
        A = model(X)
        left_err = loss(A, C)

        prev_size = 0
        for i, layer in enumerate(model.layers):
            size = layer.W.size
            layer.W -= eps * d[prev_size:prev_size + size].reshape(layer.W.shape)
            prev_size += size
        A = model(X)
        right_err = loss(A, C)

        err = left_err - right_err
        no_grad__abs_err: float = np.linalg.norm(err)
        loss.backward(model, X, C)
        dW = np.concatenate([layer.W.grad.ravel() for layer in model.layers])
        grad_abs_err = np.linalg.norm(err - eps * (d @ dW))

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
    os.makedirs('./plots', exist_ok=True)
    grad_test_model(X, C)
    jacobian_test_W(X, C)
    jacobian_test_X(X, C)
    gradient_test(X, C)
