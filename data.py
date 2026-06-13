import numpy as np

def generate_data(n, d, num_classes):
    X = np.random.randn(n, d)
    y = np.random.randint(0, num_classes, n)
    return X, y