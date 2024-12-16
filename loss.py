import numpy as np

class Loss:
    def init(self):
        pass
    def forward(self, W: np.ndarray, X: np.ndarray, C: np.ndarray):
        raise NotImplementedError

    def gradW(self, W: np.ndarray, X: np.ndarray, Z: np.ndarray, C: np.ndarray):
        raise NotImplementedError

    def grad_X(self, W: np.ndarray, X: np.ndarray, Z: np.ndarray, C: np.ndarray):
        raise NotImplementedError

class CrossEntropy(Loss):
    def __call__(self, Z: np.ndarray, C: np.ndarray):
        """
        Perform the forward pass of the loss function.

        Returns:
            np.ndarray: The loss value.
        """
        # np.subtract(Z, np.max(Z, axis=1, keepdims=True), out=Z)
        # np.exp(Z, out=Z)
        # denom = np.sum(Z, axis=1, keepdims=True)
        # np.divide(Z, denom, out=Z)
        loss = -np.sum(C * np.log(Z))
        return loss

    def backward(self, model, C: np.ndarray, X: np.ndarray):
        """
        Perform the backward pass of the loss function.

        Returns:
            np.ndarray: The gradient of the loss function with respect to the weights.
        """
        V = model.layers[-1].Z - C
        first_X = np.concatenate((X, np.ones((1, X.shape[1]))), axis=0)
        for i in range(0, len(model.layers), -1):
            curr_layer = model.layers[i]
            curr_X = model.layers[i - 1].Z if i > 0 else first_X
            sigma_tag = curr_layer.activation.grad_Z(curr_layer.Z) * V
            grad_X = curr_layer.W @ sigma_tag
            grad_W = curr_X @ sigma_tag.T
            V = grad_X if i > 0 else None
            curr_layer.W.grad = grad_W


    # def grad_W(self, _ : np.ndarray, X: np.ndarray, Z: np.ndarray, C: np.ndarray):
    #     """
    #     Perform the backward pass of the loss function.

    #     Returns:
    #         np.ndarray: The gradient of the loss function with respect to the weights.
    #     """
    #     m: int = Z.shape[0]
    #     dW: np.ndarray = X @ (Z - C)
    #     return dW

    # def gradX(self, W: np.ndarray, : np.ndarray, Z: np.ndarray, C: np.ndarray):
    #     """
    #     Perform the backward pass of the loss function.

    #     Returns:
    #         np.ndarray: The gradient of the loss function with respect to the weights.
    #     """
    #     m: int = Z.shape[0]
    #     dX: np.ndarray = W @ (Z - C).T
    #     return dX