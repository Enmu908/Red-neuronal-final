import time
import tracemalloc
import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def measure_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    return result, end - start

def measure_memory(func, *args):
    tracemalloc.start()
    result = func(*args)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak