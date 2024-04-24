from nnfs.datasets import spiral_data
import numpy as np

# This script implements a simple Neural Network using classes to represent layers and Activation Functions.

# X = Training dataset
X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


class Layer_Dense:
    # Initizalize Layers
    def __init__(self, n_inputs, n_neurons):
        # random.randn method creates a matrix of random numbers with dimensions/shape of (n_inputs, n_neurons). It's typically used to initialize weight values, and number of neurons in a neural network.
        self.weights = np.random.randn(n_inputs, n_neurons)
        # zeros method creates a matrix of zeros with dimensions/shape of (1, n_neurons). It's typically used to initialize bias values in a neural network.
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# Loss_CategoricalCrossEntropy extends Loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    
# When you save a model you are saving the weights and the biases. When you load the model this is all you are doing.

X, y = spiral_data(samples=100, classes=3)

# Instaintiating Hidden Layers
# Layer 1
layer1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

# Layer 2
layer2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Layer 1
layer1.forward(X)
activation1.forward(layer1.output)
print(str(activation1.output) + 'Layer1\n')

# Layer 2
layer2.forward(activation1.output)
activation2.forward(layer2.output)
print('\n' + str(activation2.output[:5]) + 'Layer2 First 5\n')

losss_function = Loss_CategoricalCrossEntropy()
loss1 = losss_function.calculate(activation2.output, y)


print('Loss: ', loss1)