import numpy as np
from SGD import SGD, SGD_momentum
from activations import Activation

class Layer: #Dvir

    def __init__(self, input_len, output_len ,activation : Activation, w_init = None):
        """Initialize the layer
        Args:
            input_len (int): input length of the layer
            output_len (int): output length of the layer
            activation (Activation): activation function of the layer
            w_init (np.ndarray): initial weights of the layer
            solver (Solver): solver for the layer
        """
        self.activation = activation
        self.Z = None
        self._grad_x = None
        self._grad_w = None
        self.W = w_init if w_init else np.random.rand(input_len, output_len) # might need to allow different randomization

    def forward(self, X):
        """Forward pass of the layer
        Args:
            X (np.ndarray): input of the layer
        """
        out = self.activation(self.W, X)
        self.Z = out
        return out

    def backward(self, X, V, C = None):
        """Backward pass of the layer
        Args:
            X (np.ndarray): input of the layer
            V (np.ndarray): accumulated gradient from the next layer
        """
        print(f"{X.shape=}, {V.shape=}, {self.W.shape=}, {self.Z.shape=}")
        if C is None:
            activation_grad_x = self.activation.grad_x(X, self.W, self.Z)
            activation_grad_w = self.activation.grad_w(X, self.W, self.Z)
        else:
            activation_grad_x = self.activation.grad_x(X, self.W, self.Z, C)
            activation_grad_w = self.activation.grad_w(X, self.W, self.Z, C)
        print(f"{activation_grad_x.shape=}")
        print(f"{activation_grad_w.shape=}")

        grad_mul_V_x = activation_grad_x * V
        print(f"{grad_mul_V_x.shape=}")

        grad_x = np.dot(grad_mul_V_x, self.W.T)
        print(f"{grad_x.shape=}")


        grad_mul_V_w = activation_grad_w * V
        print(f"{grad_mul_V_w.shape=}")

        grad_w = np.dot(X.T, grad_mul_V_w)
        print(f"{grad_w.shape=}")
        # else:
        #     print(f"{C.shape=}")
        #     m = self.Z.shape[0]
        #     loss = (1 / m) * (self.Z - C)
        #     grad_x = np.dot(loss, self.W.T)
        #     print(f"{grad_x.shape=}")
        #     grad_w = np.dot(X.T, loss)
        #     print(f"{grad_w.shape=}")

        self._grad_x = grad_x # accumlated grad_x
        self._grad_w = grad_w # accumlated grad_w with later grad_x


    def __call__(self, X):
        return self.forward(X)

    @property
    def weights(self):
        """Return the weights of the layer"""
        return self.W
    @property
    def grad_x(self):
        """Return the gradient of the loss with respect to the input"""
        return self._grad_x

    @property
    def grad_w(self):
        """Return the gradient of the loss with respect to the input"""
        return self._grad_w

    def update_weights(self, solver):
        """Update the weights of the layer"""
        self.W = solver.update(self.W, self._grad_w)



class ResNetLayer(Layer):

    def __init__():
        pass

    def __forward__():
        pass

    def __backward__():
        pass


if __name__ == "__main__":
    #NN
    Layer1 = Layer(np.random.rand(3, 3), ReLU())
    Layer2 = Layer(np.random.rand(3, 3), ReLU())
    Layer3 = Layer(np.random.rand(3, 3), SoftMax())
    
    forward1 = Layer1(np.random.rand(3, 3))
    forward2 = Layer2(forward1)
    forward3 = Layer3(forward2)
    print(forward3)
    
    Layer3.backward(forward3, prev_grad= np.array([1, 1, 1]))
    grad_accumlation = Layer3.grad_x
    Layer2.backward(forward2, prev_grad=grad_accumlation)
    grad_accumlation = np.dot(Layer2.grad_x.T, grad_accumlation)
    Layer1.backward(forward1, prev_grad=grad_accumlation)
