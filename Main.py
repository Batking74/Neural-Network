import numpy as np
# Calculating the output of multiple Neurons in a Neural Network using custom algorithm
inputs1 = [1, 2, 3, 2.5]

# Neurons
weights1 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Biases
biases1 = [2, 3, 0.5]

# Outputs
outputs = [0, 0, 0]

# Summation
for i in range(len(inputs)):
    outputs[0] += inputs1[i] * weights1[0][i]
    outputs[1] += inputs1[i] * weights1[1][i]
    outputs[2] += inputs1[i] * weights1[2][i]

print(outputs)





# Calculating the output of multiple Neurons in a Neural Network using dot method in more efficent way
inputs2 = [1, 2, 3, 2.5]

# Neurons
weights2 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Biases
biases2 = [2, 3, 0.5]

# Summation
output = np.dot(weights2, inputs2) + biases2

print(output)