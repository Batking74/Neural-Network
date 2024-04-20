# Calculating the output of a single Neuron in a Neural Network.
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = 0

# Summation
for i in range(len(inputs)):
    output += inputs[i] * weights[i]

print(output + bias)