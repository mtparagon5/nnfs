
import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

# adding a second layer

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1.0, 2.0, -0.5]

# if inputs is vector, np.dot(weights, inputs) will work
# if inputs is a matrix, want to transpose weights and will put inputs first -> (inputs, weights.T)
# basic matrix multiplication -> row[0] * col[0]

# layer 1
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# layer 2
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)


### batching ###
# allows calculations in parallel


# as an object -> see 4b.py
