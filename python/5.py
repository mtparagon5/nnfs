### Activation Functions ###
# step function -> 0 or 1 output
# sigmoid function -> more granular output
# Rectified Linear Unit -> ReLU(x) -> 0 if <= 0, else x if > 0  -- popular for hidden layers in neural networks

# # example activation function
# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []
# for i in inputs:
#     # if i > 0:
#     #     output.append(i)
#     # else:
#     #     output.append(0)
#     output.append(max(0, i))
# print(output)

# why activation functions?
# if only weights and biases, result is a linear function (mx+b) - output is linear - can only fit linear functions well
# non-linear activation functions allow the fitting of other functions, e.g., sigmoid function

# ReLUs are less complex as sigmoid activation functions - can be as good and less costly

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# data inputs
X, y = spiral_data(100, 3)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # by putting inputs first, won't have to transpose later
        self.weights = 0.10 * \
            np.random.randn(n_inputs, n_neurons)  # to scale -1 to 1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# layer 1
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)

# replace values < 0 with 0 (activation layer)
activation1.forward(layer1.output)

# print(layer1.output)
print(activation1.output)
