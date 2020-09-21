# This is the start of our notepad for testing the models. If you are pulling from my repo, you might need to change where the knn import points to.
from sklearn.neighbors import KNeighborsClassifier
from knn import *  # Pull in our model
from sklearn.datasets import load_iris  # Test data from the sklearn directory.
# This is an easy enough function to replicate, but this is cleaner.
from sklearn.model_selection import train_test_split


# We will load in the toy data, and split it up so that its in a useable format for our models.
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=42, test_size=0.2)

# We instantiate my model which uses only numpy.
testing_neighbors = KNN(k=3)

# We fit the model to the training data.
testing_neighbors.fit(X_train, y_train)

# We run some predictions using the test sample data.
prediction = testing_neighbors.predict(X_test)

# We score the prediction accuracy.
rob_pred = testing_neighbors.score(X_test, y_test)

print('The accuracy of the "No SKLEARN model" is :', rob_pred)
# Our model comes in at 96.667% accurate. The leakage here is by using a scratch Euclidean distance generator rather than the spatial one
# found in scipy. This is an acceptable difference for the scratch benefits.


# The process is the sample for the true model. First we instantiate it.
neigh = KNeighborsClassifier(n_neighbors=3)

# Then we fit the training data.
neigh.fit(X_train, y_train)

# Then we find its score and compare the too.
actual_preds = neigh.score(X_test, y_test)


print("The accuracy of the SKLEARN model is:", actual_preds)
# The sklearn model comes in with a 100% accuracy, which is understandable.
