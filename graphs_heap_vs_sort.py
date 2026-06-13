import numpy as np
import heapq
import time
import matplotlib.pyplot as plt

sizes = [1000, 5000, 10000, 50000, 100000]

k = 10

sort_times = []
heap_times = []

for n in sizes:

    arr = np.random.rand(n)

    # SORT
    start = time.time()

    sorted(arr)[-k:]

    sort_time = time.time() - start

    # HEAP
    start = time.time()

    heapq.nlargest(k, arr)

    heap_time = time.time() - start

    sort_times.append(sort_time)
    heap_times.append(heap_time)

# gráfica
plt.figure(figsize=(10, 6))

plt.plot(sizes, sort_times, marker='o', label='Sort')

plt.plot(sizes, heap_times, marker='o', label='Heap')

plt.xlabel("n")

plt.ylabel("Tiempo (s)")

plt.title("Heap vs Sort")

plt.legend()

plt.grid(True)

plt.show()