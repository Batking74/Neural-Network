import math

# This script calculates the output of multiple Neurons in a Neural Network using a custom algorithm and applies the sigmoid activation function to the outputs.

# Training Data
inputs1 = [1, 2, 3, 2.5]

# Neurons
# Calculating the output of multiple Neurons in a Neural Network using custom algorithm
weights1 = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]


# Activation Function
def sigmoid(x):
    return 1 / (1 + math.e ** -x)


def initForwardPropagation(x, w):
    # Outputs
    outputs = [0, 0, 0]
    # Biases
    biases1 = [2, 3, 0.5]
    
    # Summation/Forward Propagation
    for i in range(len(x)):
        outputs[0] += x[i] * w[0][i]
        outputs[1] += x[i] * w[1][i]
        outputs[2] += x[i] * w[2][i]
        
    outputs[0] += biases1[0]
    outputs[1] += biases1[1]
    outputs[2] += biases1[2]
    
    # Apply activation function (sigmoid)
    outputs = [sigmoid(output) for output in outputs]
    return outputs

probabilityDistribution = initForwardPropagation(inputs1, weights1)

print(probabilityDistribution)