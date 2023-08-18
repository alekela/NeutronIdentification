import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

with open('data_w_source.txt') as f:
    data = f.readlines()
data = list(map(lambda x: list(map(float, x.split())), data))
"""plt.plot(range(0, len(data[0])), data[0])
plt.show()"""
params = []
for sample in data:
    i = sample.index(max(sample))
    params.append((sum(sample[i - 10: i + 250]), sum(sample)))
with open("class_labels.txt") as f:
    labels = f.read()
labels = list(map(int, labels.split()))

df = pd.DataFrame({'sum around max': list(map(lambda x: x[0], params)), 'all sum': list(map(lambda x: x[1], params)),
                   'label': labels})
# print(df)
df['sum around max'] /= max(df['sum around max'])
df['all sum'] /= max(df['all sum'])
df1 = df[df['label'] == 1]
df0 = df[df['label'] == 0]
# plt.scatter(df0['sum around max'], df0['all sum'])
# plt.scatter(df1['sum around max'], df1['all sum'])
# plt.show()

X = df[['sum around max', 'all sum']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"Accuracy = {round(knn.score(X_test, y_test) * 100, 2)}%")
print(sklearn.metrics.confusion_matrix(y_test, knn.predict(X_test)))

# accuracy from num of neighbors
"""accs = []
for i in range(1, 15):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    accs.append(knn.score(X_test, y_test))

plt.plot(range(1, 15), accs)
plt.show()"""

# visualization
"""grid = np.meshgrid(np.linspace(-0.1, 1, 500), np.linspace(-0.1, 1, 500))
tmp1 = []
tmp2 = []
for i in grid[0]:
    tmp1.extend(i)
for i in grid[1]:
    tmp2.extend(i)
X_new = pd.DataFrame({'sum around max': tmp1, 'all sum': tmp2})
X_new['label'] = knn.predict(X_new)
df1_new = X_new[X_new['label'] == 1]
df0_new = X_new[X_new['label'] == 0]
plt.scatter(df0_new['sum around max'], df0_new['all sum'])
plt.scatter(df1_new['sum around max'], df1_new['all sum'])

plt.scatter(df0['sum around max'], df0['all sum'])
plt.scatter(df1['sum around max'], df1['all sum'])
plt.show()"""
