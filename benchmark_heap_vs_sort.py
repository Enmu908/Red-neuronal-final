import numpy as np
import heapq
import time

# tamaños a probar
sizes = [1000, 5000, 10000, 50000, 100000]

k = 10

print("\n=== BENCHMARK HEAP VS SORT ===\n")

for n in sizes:

    arr = np.random.rand(n)

    # -----------------------------
    # SORT COMPLETO
    # -----------------------------
    start = time.time()

    top_sort = sorted(arr)[-k:]

    sort_time = time.time() - start

    # -----------------------------
    # HEAP
    # -----------------------------
    start = time.time()

    top_heap = heapq.nlargest(k, arr)

    heap_time = time.time() - start

    # -----------------------------
    # RESULTADOS
    # -----------------------------
    print(f"n = {n}")

    print(f"Sort Time : {sort_time:.6f}s")

    print(f"Heap Time : {heap_time:.6f}s")

    print("-" * 40)