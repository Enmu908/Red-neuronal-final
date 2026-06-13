import numpy as np
import heapq

def quickselect(arr, k):
    if len(arr) == 1:
        return arr[0]

    pivot = arr[len(arr)//2]

    lows  = [x for x in arr if x < pivot]
    highs = [x for x in arr if x > pivot]
    pivots = [x for x in arr if x == pivot]

    if k < len(lows):
        return quickselect(lows, k)
    elif k < len(lows) + len(pivots):
        return pivot
    else:
        return quickselect(highs, k - len(lows) - len(pivots))


def top_k_indices(losses, k):
    heap = []

    for i, loss in enumerate(losses):
        if len(heap) < k:
            heapq.heappush(heap, (loss, i))
        else:
            if loss > heap[0][0]:
                heapq.heapreplace(heap, (loss, i))

    return [idx for (_, idx) in heap]


class HardNegativeMining:
    def __init__(self, k):
        self.k = k

    def compute_losses(self, y_pred, y_true):
        m = y_pred.shape[0]
        return -np.log(y_pred[np.arange(m), y_true] + 1e-9)

    def select_hard_examples(self, losses):
        return top_k_indices(losses, self.k)
    

# Margen de precision formula Aciertos/Total de datos