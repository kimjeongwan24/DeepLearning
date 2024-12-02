import numpy as np
import nnfs
from nnfs.datasets import spiral_data


class Layer_Dense:
    def __init__(self, n_inputs, n_outputs):
        """
        :param n_inputs: 입력의 개수
        :param n_outputs: 뉴런의 개수 == 출력의 개수
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))

    def forward(self, inputs):
        """
        :param inputs: 입려된 값
        """

        # y = ax + b
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues): #3개의 사용 위치(3개의동적)
        """
        :param dvalues: 앞선 미분 값
        """
        #f(x,w) = xw
        # w 편미분 f'(x,w) = x
        self.dweights = np.dot(self.weights.T,dvalues)
        # f(x,y) = x + y
        # y 편미분 = 1
        self.dbiases = np.sum(dvalues, axis=0, keepdims =True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        # 0보다 클 때 입력값이 그대로

    def backward(self, dvalues): #3개의 사용 위치(3개의 동적)
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        # self.inputs의 값이 0보다 작을 때는 0으로 만든다



class Activation_Softmax:
    def forward(self, inputs):
        """
        :param inputs: 입력
        """
        self.inputs = inputs

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        #각각의 입력에 대해 노멀라이즈 0~1 값으로 변경
        self.output = probabilities

    def backward(self, dvalues): #3개의 사용 위치(3개의 동적)
        """
        :param dvalues: 이전의 미분값
        """
        self.dinputs = np.empty_like(dvalues) #이전의 미분값을 받았을 때 그 형태와 동일한 빈 배열
        # enumerate zip
        # enumerate는 index랑 내부의 값을 순차적으로 내뱉는 역할
        # enumerate(['A','B','C'])
        # 0,'A'            1,'B'         2,'C'
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            #reshape -1,1
            # (2,2) => 4,1  (10,20) => 200,1
            # output 결과를 펼침
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses) #오차들의 평균을 구하기 위해
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples), y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, y_pred, y_true):
        self.activation.forward(y_pred)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3,3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

X,y = spiral_data(samples = 100, classes = 3)

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

print('loss', loss)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)

accuracy = np.mean(predictions == y)

print('accuracy', accuracy)

loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

print(dense1.dweights)