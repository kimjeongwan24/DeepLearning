import numpy as np

inputs = [[1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 4.0, 5.0, 6.0, 7.0],]

weights = [[0.2, 0.4, 0.8, 1.0, 1.2],
           [0.4, 0.8, 1.0, 1.2, 1.4],
           [0.8, 1.0, 1.2, 1.4, 1.6]]


biases = [2.0,3.0,0.5]

layers_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layers_outputs)

weights = [[0.2, 0.4, 0.8],
           [0.4, 0.8, 1.0],
           [0.8, 1.0, 1.2]]

biases = [2.0,3.0,0.5]

layers_outputs = np.dot(layers_outputs, np.array(weights).T) + biases
print(layers_outputs)
