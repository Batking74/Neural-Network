import numpy as np

# Training Data
inputs1 = [1, 2, 3, 2.5]

# Each array inside of the weights1 array is a Neuron that holds weights
weights1 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Bias for each Neuron
biases1 = [2, 3, 0.5]

# Summation/Forward Propagation (dot method does this process in a more efficent way)
output = np.dot(weights1, inputs1) + biases1
print('Output for Summation using dot Method: ', output)