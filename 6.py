import numpy as np  # 행렬을 쉽게 쓰기 위해
import nnfs  ## 데이터 호출 및 random seed 고정
from nnfs.datasets import vertical_data

# Dense Layer
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """
        :param n_inputs:  입력의 개수
        :param n_neurons: 출력의 개수
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        :param inputs: 입력
        :return: 뉴런 연산의 결과 y = ax+b
        """
        return np.dot(inputs, self.weights) + self.biases


# Relu Activation
class Activation_ReLU:
    def forward(self, inputs):
        """
        :param inputs: 뉴런의 출력
        :return: activation 결과
        """
        return np.maximum(0, inputs)


# Softmax Activation
class Activation_Softmax:
    def forward(self, inputs):
        """
        :param inputs: 뉴런의 출력
        :return: activation 결과
        """
        exp = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        prob = exp / np.sum(exp, axis=1, keepdims=True)
        return prob


# 손실 함수 클래스
class Loss:
    def calculate(self, output, y):
        """
        :param output: Dense의 출력  Dense + activation 한 결과
        :param y: 정답지의 출력 실제 정답지
        :return: 결과와 정답지의 차이 (단, 우리가 정의한 식으로 계산됨)
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


# 카테고리 손실 함수 클래스
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)  # clip 하는 이유 : log때문에 무한대로 발산하지 않게

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# 데이터 생성
X, y = vertical_data(samples=100, classes=10)  # 클래스 수를 10으로 설정

# 레이어 및 활성화 함수 초기화
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 10)  # 출력 뉴런이 10개여야 함 (클래스가 10개이므로)
activation2 = Activation_Softmax()

# 손실 함수
loss_function = Loss_CategoricalCrossentropy()

# 최적화 변수 설정
lowest_loss = 9999999  # 초깃값을 매우 크게 설정
best_dense1_weights = dense1.weights.copy()
best_dense2_weights = dense2.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_biases = dense2.biases.copy()

# 학습 반복문
for iteration in range(100000):
    # 가중치와 편향에 무작위 변동 추가
    dense1.weights += 0.01 * np.random.randn(2, 3)
    dense2.weights += 0.01 * np.random.randn(3, 10)  # 출력 뉴런에 맞게 수정
    dense1.biases += 0.01 * np.random.randn(1, 3)
    dense2.biases += 0.01 * np.random.randn(1, 10)

    # 순전파 계산
    out = activation1.forward(dense1.forward(X))
    out = activation2.forward(dense2.forward(out))

    # 손실 계산
    loss = loss_function.calculate(out, y)

    # 정확도 계산
    predictions = np.argmax(out, axis=1)
    accuracy = np.mean(predictions == y)

    # 최소 손실 갱신 시 가중치 저장
    if loss < lowest_loss:
        print(f"새로운 최적 가중치 발견! 반복: {iteration}, 손실: {loss}, 정확도: {accuracy}")
        best_dense1_weights = dense1.weights.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        # 이전 최적 가중치 복원
        dense1.weights = best_dense1_weights.copy()
        dense2.weights = best_dense2_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.biases = best_dense2_biases.copy()
