from sklearn.datasets import load_iris

iris = load_iris()

import pandas as pd 


dataset = pd.DataFrame(data=iris['data'], columns= ['sepal length', 'sepal width', 'petal length', 'petal width'])

dataset['label'] = iris['target']

print(dataset.head())

X = dataset.drop('label', axis=1)
y = dataset['label']

from sklearn.preprocessing import MinMaxScaler

x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=2)

from knn import KNearestNeighbors

knn = KNearestNeighbors(k=3)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_test, predict))

print(confusion_matrix(y_test, predict))

print(classification_report(y_test, predict))