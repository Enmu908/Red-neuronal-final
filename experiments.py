from utils import measure_time, measure_memory
from data import generate_data
import numpy as np

def stress_test_time(model, sizes, d=20, classes=3):
    results = []

    for n in sizes:
        X, y = generate_data(n, d, classes)
        _, t = measure_time(model.train_step, X, y)
        results.append((n, t))
        print(f"n={n}, tiempo={t:.4f}s")

    return results


def stress_test_memory(model_class, dims):
    results = []

    for (d, h) in dims:
        model = model_class(d, h, 3)
        _, mem = measure_memory(model.forward, np.random.randn(32, d))
        results.append((d, h, mem))
        print(f"d={d}, h={h}, memoria={mem}")

    return results