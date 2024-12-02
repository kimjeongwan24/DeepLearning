import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt
nnfs.init()
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.uniform(0,1,(n_inputs, n_neurons))
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


class Activation_ReLU:
    def forward(self, inputs):
        return np.maximum(0,inputs) #Relu

dense1 = Layer_Dense(1, 8)
dense2 = Layer_Dense(8, 8)
dense3 = Layer_Dense(8, 1)

X = np.linspace(0,2 * np.pi,100).reshape(-1,1)
y = np.sin(X)
dense1.forward(y)
dense2.forward(dense1.output)
dense3.forward(dense2.output)

plt.plot(X, y, label="True Sine Wave", color='blue')
plt.plot(X, dense3.output, label="True Activation ReLU", color='red')
plt.legend()
plt.title('Sine Wave')
plt.show()


