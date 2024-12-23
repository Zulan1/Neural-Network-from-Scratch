from typing import List
from nn import NeuralNetwork
from nn.loss import CrossEntropy
from nn.optimizer import SGD

def nn_builder(
        net_shape: List[int],
        activation: str,
        resnet: bool,
        loss: str,
        optim: str,
        lr: float,
        momentum: float
        ) -> tuple[NeuralNetwork, SGD, CrossEntropy]:
    # Initialize the model
    if resnet:
        assert len(set(net_shape[1:-1])) == 1, "ResNet layers must have the same number of units in each hidden layer"
    model = NeuralNetwork()
    L = len(net_shape) - 1
    for i in range(L):
        input_dim, output_dim = net_shape[i], net_shape[i + 1]
        if i == L - 1:
            model.add_layer(input_dim, output_dim, 'softmax')
        elif i != 0:
            model.add_layer(input_dim, output_dim, activation, resnet)
        else:
            model.add_layer(input_dim, output_dim, activation)

    # Select the optimizer
    if optim == 'sgd':
        optimizer_fn = SGD(lr, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer: {optim}")

    # Select the loss function
    if loss == 'crossentropy':
        loss_fn = CrossEntropy()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")
    
    return model, optimizer_fn, loss_fn