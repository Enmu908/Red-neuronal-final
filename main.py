from mlp import MLP
from utils import accuracy
from mnist_loader import load_mnist_like
import numpy as np

# dataset
X_train, X_test, y_train, y_test = load_mnist_like()

# modelo
mlp = MLP(64, 128, 10)

epochs = 5000
lr = 0.01

for epoch in range(epochs):

    loss = mlp.train_step(X_train, y_train, lr)

    if epoch % 100 == 0:

        pred_train = mlp.forward(X_train)
        y_pred_train = np.argmax(pred_train, axis=1)

        train_acc = accuracy(y_train, y_pred_train)

        print(
            f"Epoch {epoch} | "
            f"Loss: {loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

# evaluación final
pred = mlp.forward(X_test)

y_pred = np.argmax(pred, axis=1)

acc = accuracy(y_test, y_pred)

print("\nAccuracy final:", acc)