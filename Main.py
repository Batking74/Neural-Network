# Calculating the output of a single Neuron in a Neural Network.
inputs = [1, 2, 3, 2.5]

# Neuron 1
weight1 = [0.2, 0.8, -0.5, 1.0]
bias1 = 2

# Neuron 2
weight2 = [0.5, -0.91, 0.26, -0.5]
bias2 = 3

# Neuron 3
weight3 = [-0.26, -0.27, 0.17, 0.87]
bias3 = 0.5

# Output
output1 = 0
output2 = 0
output3 = 0

# Summation
for i in range(len(inputs)):
    output1 += inputs[i] * weight1[i]
    output2 += inputs[i] * weight2[i]
    output3 += inputs[i] * weight3[i]

print(output1 + bias1)
print(output2 + bias2)
print(output3 + bias3)