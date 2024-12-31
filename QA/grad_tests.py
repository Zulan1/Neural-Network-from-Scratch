import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat

from nn.loss import CrossEntropy
from nn.layer import Layer, ResNetLayer
from nn.activations import Tanh, SoftMax
from nn.utils import pad
from nn.utils.tensor import Tensor
from utils import nn_builder


def plot_test(grad_errs, no_grad_errs, title):
    n_iter = len(grad_errs)
    log_no_grad_errs = np.log(no_grad_errs)
    log_grad_errs = np.log(grad_errs)

    slope_no_grad, _ = np.polyfit(range(n_iter), log_no_grad_errs, 1)
    slope_grad, _ = np.polyfit(range(n_iter), log_grad_errs, 1)


    plt.plot(range(n_iter), no_grad_errs, label='No grad')#, color='b')
    plt.plot(range(n_iter), grad_errs, label='Grad')#, color='r')
    plt.text(n_iter - 2, no_grad_errs[-1], f"Slope: {slope_no_grad:.2f}")#, color='b')
    plt.text(n_iter - 2, grad_errs[-1], f"Slope: {slope_grad:.2f}")#, color='r')
    plt.yscale('log')
    plt.legend()
    plt.title(title)
    plt.savefig(f'./figures/{title}.png')
    plt.show()

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
    
    plot_test(grad_errs, no_grad_errs, 'Softmax Gradient Test')

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
    
    plot_test(grad_errs, no_grad_errs, 'Tanh W Jacobian Test')

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
    
    plot_test(grad_errs, no_grad_errs, 'Tanh X Jacobian Test')

def grad_test_model(X, C, resnet=False):
    net_shape = [X.shape[0], 10, 10, C.shape[0]]

    model, _, loss = nn_builder(net_shape,
                                activation='tanh',
                                resnet=resnet,
                                loss='crossentropy',
                                optim='sgd',
                                lr=1e-2,
                                momentum=0,
                                )
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
    
    plot_test(grad_errs, no_grad_errs, f'Model Gradient Test {"ResNet" if resnet else ""}')

def resnet_jacobian_test_W1(X, C):
    in_shape = X.shape[0]
    out_shape = X.shape[0]
    layer = ResNetLayer(in_shape, out_shape, Tanh())
    W1: Tensor = layer.W1.view()
    d = np.random.randn(*W1.shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    W2: Tensor = layer.W2.view()
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        layer.W1 += eps * d
        layer(X)
        left_A = layer.A.copy()

        # Compute right term
        layer.W1 -= eps * d
        layer(X)
        right_A = layer.A.copy()

        err = left_A - right_A
        dA = layer.activation.grad(layer.inner_A)
        diff = eps * W2 @ ((d.T @ pad(X)) * dA)
        no_grad_abs_err: float = np.linalg.norm(err)
        grad_abs_err = np.linalg.norm(err - diff)
        no_grad_errs.append(no_grad_abs_err)
        grad_errs.append(grad_abs_err)

    
    plot_test(grad_errs, no_grad_errs, 'ResNet W1 Jacobian Test')

def resnet_jacobian_test_W2(X, C):
    in_shape = X.shape[0]
    out_shape = X.shape[0]
    layer = ResNetLayer(in_shape, out_shape, Tanh())
    W2: Tensor = layer.W2.view()
    d = np.random.randn(*W2.shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []

    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute left term
        layer.W2 += eps * d
        layer(X)
        left_A = layer.A.copy()

        # Compute right term
        layer.W2 -= eps * d
        layer(X)
        right_A = layer.A.copy()

        err = left_A - right_A
        diff = eps * d @ layer.inner_A
        no_grad_abs_err: float = np.linalg.norm(err)
        grad_abs_err = np.linalg.norm(err - diff)
        no_grad_errs.append(no_grad_abs_err)
        grad_errs.append(grad_abs_err)

    
    plot_test(grad_errs, no_grad_errs, 'ResNet W2 Jacobian Test')

def resnet_jacobian_test_X(X, C):
    in_shape = X.shape[0]
    out_shape = X.shape[0]
    layer = ResNetLayer(in_shape, out_shape, Tanh())
    W1: Tensor = layer.W1.view()
    W2: Tensor = layer.W2.view()
    d = np.random.randn(*X.shape)
    d /= np.linalg.norm(d)
    n_iter = 10
    grad_errs = []
    no_grad_errs = []

    for i in range(1, n_iter + 1):
        eps = 10 * (0.5 ** i)

        # Compute left term
        left_A = layer(X + eps * d)

        # Compute right term
        right_A = layer(X)

        err = left_A - right_A
        dA = layer.activation.grad(layer.inner_A)
        diff = eps * (d + W2 @ (dA * (W1[:-1, :].T @ d)))
        no_grad_abs_err: float = np.linalg.norm(err)
        grad_abs_err = np.linalg.norm(err - diff)
        no_grad_errs.append(no_grad_abs_err)
        grad_errs.append(grad_abs_err)

    
    plot_test(grad_errs, no_grad_errs, 'ResNet X Jacobian Test')


if __name__ == '__main__':
    Data = loadmat('Data/GMMData.mat')
    X, C = Data['Yt'], Data['Ct']
    os.makedirs('./figures', exist_ok=True)
    resnet_jacobian_test_X(X, C)
    resnet_jacobian_test_W1(X, C)
    resnet_jacobian_test_W2(X, C)
    grad_test_model(X, C, resnet=True)
    jacobian_test_W(X, C)
    jacobian_test_X(X, C)
    gradient_test(X, C)
