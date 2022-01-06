import numpy as np
from random import random


def sigmoid(x):
    # Функция активации: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


class Neuron:
    def __init__(self, lay, pos):
        self.lay_number = lay
        self.number = pos

        self.amount = 0
        self.value = sigmoid(self.amount)
        self.bias = 0

        self.weights = []

    def count_value(self, weights, previous_layer, num_weights):
        self.weights = weights
        tmp = 0
        for i in range(num_weights):
            j = num_weights * self.number + i
            tmp += weights[j] * previous_layer.neurons[j].value
        self.amount = tmp + self.bias
        self.value = sigmoid(self.amount)
        return self.value


class FirstNeuron(Neuron):

    def count_value(self):
        self.value = sigmoid(self.amount)
        return self.value

    def set_amount(self, data):
        self.amount = data
        self.count_value()


class Layer:
    def __init__(self, number_of_layer, number_of_neurons):
        self.neurons = [Neuron(number_of_layer, q) for q in range(number_of_neurons)]

    def count_values(self, weights, previous_layer):
        for q in range(len(self.neurons)):
            self.neurons[q].count_value(weights[q], previous_layer,
                                        int(len(previous_layer.neurons) / len(self.neurons)))

    def __len__(self):
        return len(self.neurons)


class FirstLayer(Layer):
    def __init__(self, number_of_layer, number_of_neurons):
        self.neurons = [FirstNeuron(number_of_layer, q) for q in range(number_of_neurons)]

    def set_data(self, data):
        if len(self.neurons) != len(data):
            raise Exception("Количество данных должно совпадать с количеством нейронов первого слоя!")
        for i in range(len(data)):
            self.neurons[i].set_amount(data[i])

    def count_values(self, weights, previous_layer):
        pass

    def __len__(self):
        return len(self.neurons)


class OurNeuralNetwork:

    def __init__(self, num_of_layers, *number_of_neurons_in_layers):
        """инициализация нейронки, количество слоёв и количество нейронов в каждом слое"""
        self.layers = [FirstLayer(0, number_of_neurons_in_layers[0])]
        for i in range(1, num_of_layers):
            self.layers.append(Layer(i, number_of_neurons_in_layers[i]))

        self.weights = []
        try:
            with open("weights.txt") as f:
                data = f.read().split("__\n")[:-1]
                for layer_weights in data:
                    layer_weights = layer_weights.split('\n')[:-1]
                    for num in range(len(layer_weights)):
                        layer_weights[num] = list(map(float, layer_weights[num].split()))
                    self.weights.append(layer_weights)

        except FileNotFoundError:
            self.create_new_weights()

        try:
            with open("biases.txt") as f:
                data = f.read().split("\n")
                if len(data) > 1:
                    for i in range(len(data)):
                        data[i] = list(map(float, data[i].split()))
                        for j in range(len(data[i])):
                            self.layers[i].neurons[j].bias = data[i][j]

        except FileNotFoundError:
            self.create_new_biases()

    def create_new_weights(self):
        self.weights = [[]]
        for i in range(1, len(self.layers)):
            q = int(len(self.layers[i - 1]) / len(self.layers[i]))
            prev = len(self.layers[i - 1])
            s = []
            for j in range(len(self.layers[i])):
                s.append([0 for _ in range(q * j)] + [random() for _ in range(q)] +
                         [0 for _ in range(prev - q - q * j)])

            self.weights.append(s)

    def create_new_biases(self):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].neurons)):
                self.layers[i].neurons[j].bias = random()

    # просто функция, закидываешь данные, возвращается ответ
    def feedforward(self, data):
        self.layers[0].set_data(data)
        for q in range(1, len(self.layers)):
            self.layers[q].count_values(self.weights[q], self.layers[q - 1])
        return self.layers[-1].neurons[0].value

    def get_all_neurons_values(self):
        values = []
        for i in self.layers:
            for j in i.neurons:
                values.append(j.value)
        return values

    def train(self, data, right_answers):
        learn_rate = 0.1
        epochs = 120  # количество эпох обучения

        for epoch in range(epochs):
            if epoch % 10 == 0:
                print(f"Обучено {epoch} из {epochs}")

            for x, y_true in zip(data, right_answers):

                y_pred = self.feedforward(x)

                d_ypred = -2 * (y_true - y_pred)

                for i in range(1, len(self.layers)):
                    for now in range(len(self.layers[i])):
                        iters = int(len(self.layers[i - 1]) / len(self.layers[i]))
                        for pre in range(iters * now, iters * (now + 1)):
                            self.weights[i][now][pre] -= learn_rate * d_ypred * \
                                                         self.layers[i - 1].neurons[pre].value * \
                                                         deriv_sigmoid(self.layers[i].neurons[now].amount)

                for i in range(len(self.layers)):
                    for neuron in self.layers[i].neurons:
                        neuron.bias -= learn_rate * d_ypred * deriv_sigmoid(neuron.amount)


"""
network = OurNeuralNetwork(3, 4, 2, 1)
print(network.feedforward([1, 2, 3, 4]))
"""  # пример использования нейронки: в нашем случае в последнем слое должен быть 1 нейрон
