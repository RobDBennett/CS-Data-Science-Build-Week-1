from knn import *
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42, test_size=.2)

clf = KNN(3)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

for i in prediction:
    print(i, end= ' ')

print(prediction == y_test)
print(clf.score(X_test, y_test))