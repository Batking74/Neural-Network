import numpy as np
import math

# Basic implementation of a Neural Network for classification tasks, utilizing ReLU and Softmax Activation Functions.

# Training Data
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]

# Weights
W = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# Bias
B = [2, 3, 2]


# Rectified Linear Unit Activation Function
def ReLU(outputs):
    activated = []
    for output in outputs:
        if(output > 0):
            activated.append(output)
        else:
            activated.append(0)
    return activated


def normalizeExpValues(exp_values):
    exp_sum = 0
    normalized_values = []
    for exp_val in exp_values:
        exp_sum += exp_val

    for exp_val in exp_values:
        normalized_values.append(exp_val / exp_sum)
    return normalized_values


def softmaxActivation(x):
    # Exponential Funtion: math.e = Euler's Number
    exp_values = [math.e ** i for i in x]
    return normalizeExpValues(exp_values)


# Forward Propagation
def forwardPropagation():
    outputs = [0, 0, 0]
    for i in range(len(X)):
        for j in range(len(X[0])):
            outputs[i] += W[i][j] * X[i][j]
        outputs[i] += B[i]
    outputs = ReLU(outputs)
    outputs = softmaxActivation(outputs)
    return outputs

output = forwardPropagation()

print(output)





# Extra Help
softmax_outputs = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.5, 0.4],
    [0.02, 0.9, 0.08],
])

# class meaning: Dog, Cat, Tiger, etc
class_targets = [0, 1, 1]

# Row column
print(softmax_outputs[[0, 1, 2], class_targets])