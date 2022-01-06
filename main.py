from prepare_data import Data
from neural_network import OurNeuralNetwork
import time

inputdata = Data('data_w_source.txt', "class_labels.txt")

data = inputdata.get_data()
answers = inputdata.get_answers()

for i in range(len(data)):
    tmp = data[i].index(max(data[i]))
    data[i] = data[i][tmp - 50: tmp + 450]

start_time = time.time()

network = OurNeuralNetwork(3, 500, 25, 1)


def training():
    network.train(data, answers)
    with open("weights.txt", "w") as f:
        for i in network.weights:
            for j in i:
                f.write(" ".join(map(str, j)) + "\n")
            f.write("__\n")

    with open("biases.txt", "w") as f:
        for i in network.layers:
            s = []
            for j in i.neurons:
                s.append(j.bias)
            f.write(" ".join(map(str, s)) + "\n")
    print("Тренировка окончена")


def practic():
    prac_anss = []
    for i in range(len(data)):
        tmp = network.feedforward(data[i])
        prac_anss.append(round(tmp))
    right = 0
    lie_0 = 0
    lie_1 = 0

    for i in range(len(answers)):

        if prac_anss[i] == answers[i]:
            right += 1
        elif prac_anss[i] == 1 and answers[i] == 0:
            lie_1 += 1
        elif prac_anss[i] == 0 and answers[i] == 1:
            lie_0 += 1

    print("Нейронная сеть работала: %.2f секунд" % (time.time() - start_time))
    print(f"Процент ошибки: {right / len(prac_anss) * 100} %")
    print(f"Ложные нули: {lie_0}, {lie_0 / len(prac_anss) * 100} %")
    print(f"Ложные единицы: {lie_1}, {lie_1 / len(prac_anss) * 100} %")


practic()
