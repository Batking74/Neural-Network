import numpy as np

# This script performs forward propagation through a neural network with two layers.

# Batch of inputs/Training Data
inputs3 = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Each Layer has their own weights and bias: Layer 1
weights3 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]
biases3 = [2, 3, 0.5]


# Layer 2
weights4 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]
biases4 = [-1, 2, -0.5]

# Transposing weights Matrix then performing summation
layer1_outputs = np.dot(inputs3, np.array(weights3).T) + biases3
layer2_outputs = np.dot(layer1_outputs, np.array(weights4).T) + biases4
print('Output for Summation with multiple layers: ', layer2_outputs)