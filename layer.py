import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

# 데이터 생성 및 시각화
X, y = spiral_data(samples=100, classes=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, initialize_method='Xavier'):
        self.initialize_method = initialize_method

        # Initialize weights
        if self.initialize_method == 'Xavier':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1 / n_inputs)
        elif self.initialize_method == 'He':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        elif self.initialize_method == 'Gaussian':
            self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        else:
            raise ValueError(f"Unknown initialization method: {self.initialize_method}")

        self.biases = np.random.uniform(0, 1, (1, n_neurons))

    def forward(self, inputs):
        # Linear transformation
        linear_output = np.dot(inputs, self.weights) + self.biases
        # Apply ReLU activation (0 이상은 그대로, 0 이하에는 0)
        activated_output = np.clip(linear_output, 0, None)
        return activated_output


# 첫 번째 Dense Layer (입력 2, 뉴런 5개) - Xavier 초기화
layer1_xavier = Layer_Dense(2, 5, initialize_method='Xavier')
output1_xavier = layer1_xavier.forward(X)
print("Output of Layer 1 with Xavier initialization:")
print(output1_xavier)

# 첫 번째 Dense Layer (입력 2, 뉴런 5개) - He 초기화
layer1_he = Layer_Dense(2, 5, initialize_method='He')
output1_he = layer1_he.forward(X)
print("Output of Layer 1 with He initialization:")
print(output1_he)

# 첫 번째 Dense Layer (입력 2, 뉴런 5개) - Gaussian 초기화
layer1_gaussian = Layer_Dense(2, 5, initialize_method='Gaussian')
output1_gaussian = layer1_gaussian.forward(X)
print("Output of Layer 1 with Gaussian initialization:")
print(output1_gaussian)

# 두 번째 Dense Layer를 추가하여 출력 확인
# 두 번째 Dense Layer (입력 5, 뉴런 3개) - Xavier 초기화
layer2_xavier = Layer_Dense(5, 3, initialize_method='Xavier')
output2_xavier = layer2_xavier.forward(output1_xavier)
print("Output of Layer 2 with Xavier initialization:")
print(output2_xavier)

# 두 번째 Dense Layer (입력 5, 뉴런 3개) - He 초기화
layer2_he = Layer_Dense(5, 3, initialize_method='He')
output2_he = layer2_he.forward(output1_he)
print("Output of Layer 2 with He initialization:")
print(output2_he)

# 두 번째 Dense Layer (입력 5, 뉴런 3개) - Gaussian 초기화
layer2_gaussian = Layer_Dense(5, 3, initialize_method='Gaussian')
output2_gaussian = layer2_gaussian.forward(output1_gaussian)
print("Output of Layer 2 with Gaussian initialization:")
print(output2_gaussian)