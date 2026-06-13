import numpy as np


class MLP:

    # ---------------------------------------------------
    # INICIALIZACIÓN
    # ---------------------------------------------------

    def __init__(self, input_size, hidden_size, output_size):

        # pesos capa 1
        self.W1 = np.random.randn(
            input_size,
            hidden_size
        ) * 0.01

        # bias capa 1
        self.b1 = np.zeros((1, hidden_size))

        # pesos capa 2
        self.W2 = np.random.randn(
            hidden_size,
            output_size
        ) * 0.01

        # bias capa 2
        self.b2 = np.zeros((1, output_size))

    # ---------------------------------------------------
    # ACTIVACIÓN ReLU
    # ---------------------------------------------------

    def relu(self, x):

        return np.maximum(0, x)

    # derivada ReLU
    def relu_derivative(self, x):

        return (x > 0).astype(float)

    # ---------------------------------------------------
    # SOFTMAX ESTABLE
    # ---------------------------------------------------

    def softmax(self, x):

        exp = np.exp(
            x - np.max(x, axis=1, keepdims=True)
        )

        return exp / np.sum(
            exp,
            axis=1,
            keepdims=True
        )

    # ---------------------------------------------------
    # FORWARD
    # ---------------------------------------------------

    def forward(self, X):

        # entrada → oculta
        self.Z1 = np.dot(X, self.W1) + self.b1

        self.A1 = self.relu(self.Z1)

        # oculta → salida
        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        self.Y_hat = self.softmax(self.Z2)

        return self.Y_hat

    # ---------------------------------------------------
    # LOSS
    # ---------------------------------------------------

    def compute_loss(self, y_hat, y):

        m = len(y)

        log_likelihood = -np.log(
            y_hat[np.arange(m), y] + 1e-9
        )

        loss = np.sum(log_likelihood) / m

        return loss

    # ---------------------------------------------------
    # BACKWARD
    # ---------------------------------------------------

    def backward(self, X, y, lr=0.01):

        m = X.shape[0]

        # copia predicción
        dZ2 = self.Y_hat.copy()

        # derivada cross entropy
        dZ2[np.arange(m), y] -= 1

        dZ2 /= m

        # gradientes capa 2
        dW2 = np.dot(self.A1.T, dZ2)

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        # gradiente hacia atrás
        dA1 = np.dot(dZ2, self.W2.T)

        dZ1 = dA1 * self.relu_derivative(self.Z1)

        # gradientes capa 1
        dW1 = np.dot(X.T, dZ1)

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )

        # ---------------------------------------------------
        # GUARDAR GRADIENTES
        # (IMPORTANTE PARA GRADIENT CHECK)
        # ---------------------------------------------------

        self.dW1 = dW1
        self.db1 = db1

        self.dW2 = dW2
        self.db2 = db2

        # ---------------------------------------------------
        # ACTUALIZAR PESOS
        # ---------------------------------------------------

        self.W1 -= lr * dW1
        self.b1 -= lr * db1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    # ---------------------------------------------------
    # TRAIN STEP
    # ---------------------------------------------------

    def train_step(self, X, y, lr=0.01):

        # forward
        y_hat = self.forward(X)

        # loss
        loss = self.compute_loss(y_hat, y)

        # backward
        self.backward(X, y, lr)

        return loss