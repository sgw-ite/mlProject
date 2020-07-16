import numpy as np
from math import sqrt
from collections import Counter as counter

class knn_classifier():
    def __init__(self, k):
        self.k = k
        self._X_train = None
        self._y_train = None
    def fit(self,X_train,y_train):
        self._X_train = X_train
        self._y_train = y_train
        return self
    def predict(self,X_predict):
        return np.array([self._predict(x_predict) for x_predict in X_predict])
    def _predict(x_predict):
        distances = [sqrt(np.sum((x_predict - x_train)**2)) for x_train in self._X_train]
        topk_y = [self._y_train[i] for i in argsort(distances)[:k]]
        return counter(topk_y).most_common(1)[0][0]
    def __repr__(self):
        return "kNN(k = %d)" % self.k