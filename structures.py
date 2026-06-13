from collections import deque
import numpy as np
import heapq

class BatchQueue:
    def __init__(self, X, y, batch_size):
        self.queue = deque()
        for i in range(0, len(X), batch_size):
            self.queue.append((X[i:i+batch_size], y[i:i+batch_size]))

    def has_next(self):
        return len(self.queue) > 0

    def next_batch(self):
        return self.queue.popleft()


class LossTracker:
    def __init__(self):
        self.history = {}

    def log(self, epoch, batch_id, loss):
        if epoch not in self.history:
            self.history[epoch] = {}
        self.history[epoch][batch_id] = loss

    def get_epoch_loss(self, epoch):
        return list(self.history.get(epoch, {}).values())


def prune_weights_heap(W, k):
    flat = [(abs(w), idx) for idx, w in enumerate(W.flatten())]

    heap = []
    for val, idx in flat:
        if len(heap) < k:
            heapq.heappush(heap, (val, idx))
        else:
            if val > heap[0][0]:
                heapq.heapreplace(heap, (val, idx))

    mask = np.zeros(W.size)
    for _, idx in heap:
        mask[idx] = 1

    return (W.flatten() * mask).reshape(W.shape)