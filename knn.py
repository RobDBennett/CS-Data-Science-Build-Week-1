# This is the file containing the model for our KNearestNeighbors
import numpy as np


class KNN:
    def __init__(self, k=3):
        """
        Our model is built around the number of neighbors we are looking to find. 
        Let k equal the assumed number of classifications.
        """
        self.k = k

    def fit(self, X, y):
        """
        This method will fit training data to the model. We also assert that then length
        of the training data and targets are the same, otherwise the prediction method will break.
        """
        assert len(X) == len(y)
        self.X_train = X
        self.y_train = y

    def distance(self, X1, X2):
        """
        The distance method finds the Euclidean distances between two relative points. There is 
        a cleaner way to do this using the scipy.spatial library, but for the sake of completeness we have 
        done this simple math here. The two arrays are compared, the differences between them are squared, 
        and the square root is taken. The distance defaults to 0 for the case of a same value.
        """
        X1, X2 = np.array(X1), np.array(X2)
        distance = 0
        for i in range(len(X1) - 1):
            distance += (X1[i] - X2[i]) ** 2
        return np.sqrt(distance)

    def predict(self, X_test):
        """
        Method that takes the fitted model and runs the X_test data comparing
        the Euclidean distances between each point. The values are sorted, and 
        the highest values are stored to give a prediction on targets.
        """
        sorted_output = []
        for i in range(len(X_test)):
            distances = []
            neighbors = []
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_train[j], X_test[i])
                distances.append([dist, j])
            distances.sort()
            distances = distances[0:self.k]
            for d, j in distances:
                neighbors.append(self.y_train[j])
            ans = max(neighbors)
            sorted_output.append(ans)

        return sorted_output

    def score(self, X_test, y_test):
        """
        This method takes the X_test and y_test, runs the data through the predict method.
        The number of successful guesses are summed and compared to the total number in the test data.
        """
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)
