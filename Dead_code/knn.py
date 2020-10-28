from operator import itemgetter
from point import Point

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = int(k)
        self._fit_data= []

    def fit(self, X, y):
        assert len(X) == len(y)
        self._fit_data = [(Point(coordinates), label) for coordinates, label in zip(X,y)]


    def predict(self, x):
        predicts = []
        for coordinates in x:
            predict_point = Point(coordinates)

            distances = []
            for data_point, data_label in self._fit_data:
                distances.append((predict_point.distance(data_point), data_label))
            
            distances = sorted(distances, key=itemgetter(0))[:self.k]
            predicts.append(list(max(distances, key=itemgetter(1))[1]))
        return predicts
 
"""
class KNN:
    def __init__(self, k):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    @staticmethod
    def _eculidean_distance(v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        distance = 0
        for i in range(len(v1) -1):
            distance += (v1[i] - v2[i]) **2
        return np.sqrt(distance)
    
    def predict(self, train_set, test_instance):
        distances = []
        for i in range(len(train_set)):
            dist = self._eculidean_distance(train_set[i][:-1], test_instance)
            distances.append((train_set[i], dist))
        distances.sort(key=lambda x: x[1])

        neighbors = []
        for i in range(self.k):
            neighbors.append(distances[i][0])
        
        classes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]
            if response in classes:
                classes[response] += 1
            else:
                classes[response] = 1
        
        sorted_classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        return sorted_classes[0][0]

    @staticmethod
    def evaluate(y_true, y_pred):
        n_correct = 0
        for act, pred in zip(y_true, y_pred):
            if act == pred:
                n_correct += 1
        return n_correct / len(y_true)

    
    def distance(self, X1, X2):
        distance = scipy.spatial.distance.euclidean(X1, X2)
    
    def predict(self, X_test):
        final_output = []
        for i in range(len(X_test)):
            d = []
            votes = []
            for j in range(len(self.X_train)):
                dist = self.distance(self.X_train[j], X_test[i])
                d.append([dist, j])
            d.sort()
            d = d[0:self.k]
            for d, j in d:
                votes.append(self.y_train[j])
                ans = Counter(votes).most_common(1)[0][0]
                final_output.append(ans)
        return final_output
    
    def score(self, X_test, y_test):
        predictions = self.predict(X_test)
        return (predictions == y_test).sum() / len(y_test)
    """