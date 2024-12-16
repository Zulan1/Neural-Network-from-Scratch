import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from loss import SoftmaxCrossEntropy

def gradient_test(X, C):
    n = X.shape[0]
    m = X.shape[1]
    l = C.shape[1]
    W = np.random.randn(n, l) * 1e-5
    f = SoftmaxCrossEntropy()
    d = np.random.rand(n, l)
    d /= np.linalg.norm(d)
    print(f"{np.linalg.norm(d)=}, {d=}")
    n_iter = 10
    grad_errs = []
    no_grad_errs = []
    for i in range(1, n_iter + 1):
        eps = 1e-2 * (0.5 ** i)

        # Compute the loss with and without gradient
        f.forward(W + eps * d, X, C)
        left_loss = f.loss
        Z = f.forward(W, X, C)
        right_loss = f.loss
        no_grad_err = np.abs(left_loss - right_loss)
        dW = f.grad_W(W, X, Z, C)
        grad_err = np.abs(left_loss - right_loss - eps * (d.ravel() @ dW.ravel()))

        grad_diff = eps * (d.T @ dW).sum()
        print(f"{left_loss=}, {right_loss=}, {dW=}, {grad_diff=}")

        no_grad_errs.append(no_grad_err)
        grad_errs.append(grad_err)
    
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
    X = loadmat('Data/PeaksData.mat')['Yt']
    X = np.vstack([X, np.ones(X.shape[1])])
    C = loadmat('Data/PeaksData.mat')['Ct'].T
    print(f"{X.shape=}, {C.shape=}")
    gradient_test(X, C)
