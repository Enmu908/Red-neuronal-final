import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import numpy as np
import matplotlib.pyplot as plt

from mlp import MLP

hidden_sizes = [8, 16, 32, 64, 128, 256]

memory_usage = []

print("\n=== MEMORY BENCHMARK ===\n")

for h in hidden_sizes:

    mlp = MLP(64, h, 10)

    # memoria aproximada
    total_bytes = (
        mlp.W1.nbytes +
        mlp.b1.nbytes +
        mlp.W2.nbytes +
        mlp.b2.nbytes
    )

    memory_kb = total_bytes / 1024

    memory_usage.append(memory_kb)

    print(f"h = {h}")

    print(f"Memoria: {memory_kb:.2f} KB")

    print("-" * 40)

# gráfica
plt.figure(figsize=(10, 6))

plt.plot(
    hidden_sizes,
    memory_usage,
    marker='o'
)

plt.xlabel("Neuronas ocultas")

plt.ylabel("Memoria (KB)")

plt.title("Memoria vs Neuronas Ocultas")

plt.grid(True)

plt.show()