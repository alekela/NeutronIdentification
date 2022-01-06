class Data:
    """класс, который преобразует данные из тектового файла во вложенный список"""
    def __init__(self, data_name, answers_name):
        with open(data_name) as f:
            self.d = f.read().strip().split("\n")
            for i in range(len(self.d)):
                self.d[i] = list(map(float, self.d[i].split()))

        with open(answers_name) as f:
            self.ans = list(map(int, f.read().strip().split()))

    def get_data(self):
        return self.d.copy()

    def get_answers(self):
        return self.ans.copy()


"""data = Data('data_w_source.txt', "class_labels.txt")
print(data.get_answers())"""  # пример использования класса для получения подготовленных данных
