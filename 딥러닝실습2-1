import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        :param n_inputs: 입력의 개수
        :param n_neurons: 뉴런의 개수
        """
        self.weights = np.zeros((n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

class cross_entropy:
    def forward(self, predictions, targets):
        # Clip predictions to prevent log (0)
        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        # If targets are sparse
        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            # one-hot encoding
            correct_confidences = np.sum(predictions * targets, axis=1)

        # calculate negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)

        # calculate average loss
        return np.mean(negative_log_likelihoods)

# Example usage
X = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(X)

dense1 = Layer_Dense(1, 8)
dense2 = Layer_Dense(8, 8)
dense3 = Layer_Dense(8, 1)

dense1.weights = np.array([[1.0], [0.1], [1.0], [0.1], [1.0], [0.1], [1.0], [0.1]]).T
dense1.biases = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0], [3.5]])

dense2.weights = np.random.randn(8, 8) * 0.1
dense2.biases = np.ones((1, 8)) * 0.2

dense3.weights = np.random.randn(8, 1) * 0.2
dense3.biases = np.ones((1, 1))

activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_Softmax()

dense1_output = dense1.forward(X)
activation1_output = activation1.forward(dense1_output)

dense2_output = dense2.forward(activation1_output)
activation2_output = activation2.forward(dense2_output)

dense3_output = dense3.forward(activation2_output)
activation3_output = activation3.forward(dense3_output)

plt.plot(X, y, color="blue")  # sinewave
plt.plot(X, activation3_output, color="red")  # nn.output
plt.show()

softmax_outputs = np.array([
        [0.7, 0.1, 0.2],
        [0.1, 0.5, 0.4],
        [0.2, 0.2, 0.6]
    ])
targets = np.array([0, 1, 2])

loss = cross_entropy.forward(softmax_outputs, targets)
print("Categorical Cross Entropy Loss:", loss)
