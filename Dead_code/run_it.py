from knn import *
import pandas as pd
from sklearn.datasets import load_iris

iris_dataset = load_iris()
Y_iris = iris_dataset.target
iris_dataset = pd.DataFrame(iris_dataset.data, columns= iris_dataset.feature_names)
iris_dataset = pd.concat([iris_dataset, pd.Series(Y_iris)], axis=1)
iris_dataset.rename(columns={0:'class'}, inplace=True)

def train_test_split(dataset, test_size=.25):
    n_test = int(len(dataset) * test_size)
    test_set = dataset.sample(n_test)
    train_set = []
    for ind in dataset.index:
        if ind in test_set.index:
            continue
        train_set.append(dataset.iloc[ind])
    
    train_set = pd.DataFrame(train_set).astype(float).values.tolist()
    test_set = test_set.astype(float).values.tolist()

    return train_set, test_set

train_set, test_set = train_test_split(iris_dataset)
print(len(train_set), len(test_set))

knn = KNN(k=3)
preds = []

for row in test_set:
    predictors_only = row[:-1]
    prediction = knn.predict(train_set, predictors_only)
    preds.append(prediction)

actual = np.array(test_set)[:, -1]
print(knn.evaluate(actual, preds))