import numpy as np
import math

# Softmax
def softmaxActivation():
    # Exponentiate: We do this first to convert negatives to positive without loosing meaning
    exp_values = np.exp(layer_outputs)
    # Normalize
    norm_values = exp_values / np.sum(exp_values)
    return norm_values

layer_outputs = [4.8, 1.21, 2.385]
output = softmaxActivation()

print(output + '\n')
print(sum(output) + '\n')



# Sigmoid
def sigmoid(x):
    return 1 / ((math.e ** -x) + 1)
print(sigmoid(58.8) + '\n')



# Categorical Cross-Entropy
def loss(softmax_output):
    loss = 0
    target_class = 0
    for i in range(len(softmax_output)):
        if i == target_class:
            # loss += math.log(softmax_output[i] * target_output[i])
            loss += math.log(softmax_output[i])
    return -loss

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0]

print(loss(softmax_output))



# Rectified Linear Unit Activation Function
def ReLU(outputs):
    activated = []
    for output in outputs:
        if(output > 0):
            activated.append(output)
        else:
            activated.append(0)
    return activated