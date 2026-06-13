import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import time
import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP
from mnist_loader import load_mnist_like
from utils import accuracy

# dataset
X_train, X_test, y_train, y_test = load_mnist_like()

# distintas neuronas ocultas
hidden_sizes = [8, 16, 32, 64, 128]

times = []
accuracies = []

epochs = 300

print("\n=== BENCHMARK MLP ===\n")

for h in hidden_sizes:

    print(f"Entrenando con h = {h}")

    mlp = MLP(64, h, 10)

    start = time.time()

    for epoch in range(epochs):
        mlp.train_step(X_train, y_train, lr=0.01)

    elapsed = time.time() - start

    # evaluación
    pred = mlp.forward(X_test)

    y_pred = np.argmax(pred, axis=1)

    acc = accuracy(y_test, y_pred)

    times.append(elapsed)

    accuracies.append(acc)

    print(f"Tiempo   : {elapsed:.4f}s")

    print(f"Accuracy : {acc:.4f}")

    print("-" * 40)

# ---------------------------------------------------
# GRÁFICA TIEMPO
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(hidden_sizes, times, marker='o')

plt.xlabel("Neuronas ocultas")

plt.ylabel("Tiempo (s)")

plt.title("Tiempo vs Neuronas Ocultas")

plt.grid(True)

plt.show()

# ---------------------------------------------------
# GRÁFICA ACCURACY
# ---------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(hidden_sizes, accuracies, marker='o')

plt.xlabel("Neuronas ocultas")

plt.ylabel("Accuracy")

plt.title("Accuracy vs Neuronas Ocultas")

plt.grid(True)

plt.show()