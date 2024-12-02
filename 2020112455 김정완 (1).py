inputs= [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
num_neuron = 4


output = \
    inputs[0]*weights[0] + \
    inputs[1]*weights[1] + \
    inputs[2]*weights[2] + \
    inputs[3]*weights[3] + \
    bias

print(output)

import random

def init_weights(inputs):
    weights = []
    for i in range(len(inputs)):
     weights.append(random.uniform(-1, 1))

    return weights

weights = init_weights(inputs)
print(weights)

def cal(inputs, weights, bias):
    output = sum(i * w for i, w in zip(inputs, weights)) + bias
    return output

result = cal(inputs, weights, bias)
print(result)


def cal_neuron(num_neuron, inputs):
    outputs = []

    for _ in range(num_neuron):
        weights = init_weights(inputs)
        bias = random.uniform(-1, 1)

        neuron_output = cal(inputs, weights, bias)
        outputs.append(neuron_output)

    return outputs

result = cal_neuron(num_neuron , inputs)
print(result)
