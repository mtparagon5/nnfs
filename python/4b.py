import numpy as np

np.random.seed(0)

# as an object

# inputs
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # by putting inputs first, won't have to transpose later
        self.weights = 0.10 * \
            np.random.randn(n_inputs, n_neurons)  # to scale -1 to 1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
layer2.forward(layer1.output)

# print(layer1.output)
print(layer2.output)
