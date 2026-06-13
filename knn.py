import numpy as np

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        preds = []

        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            idx = np.argsort(distances)[:self.k]
            labels = self.y_train[idx]
            pred = np.bincount(labels).argmax()
            preds.append(pred)

        return np.array(preds)