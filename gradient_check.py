import numpy as np
from mlp import MLP

# ---------------------------------------------------
# DATOS PEQUEÑOS
# ---------------------------------------------------

np.random.seed(42)

X = np.random.randn(5, 4)

y = np.array([0, 1, 1, 0, 1])

# ---------------------------------------------------
# MODELO PEQUEÑO
# ---------------------------------------------------

mlp = MLP(
    input_size=4,
    hidden_size=5,
    output_size=2
)

# ---------------------------------------------------
# FORWARD + BACKWARD
# ---------------------------------------------------

loss = mlp.train_step(X, y, lr=0.0)

# guardar gradiente analítico
analytic_grad = mlp.dW1[0, 0]

# ---------------------------------------------------
# GRADIENTE NUMÉRICO
# ---------------------------------------------------

epsilon = 1e-5

original = mlp.W1[0, 0]

# J(w + epsilon)
mlp.W1[0, 0] = original + epsilon

pred_plus = mlp.forward(X)

loss_plus = mlp.compute_loss(pred_plus, y)

# J(w - epsilon)
mlp.W1[0, 0] = original - epsilon

pred_minus = mlp.forward(X)

loss_minus = mlp.compute_loss(pred_minus, y)

# restaurar peso original
mlp.W1[0, 0] = original

# derivada numérica
numeric_grad = (
    (loss_plus - loss_minus)
    / (2 * epsilon)
)

# ---------------------------------------------------
# ERROR RELATIVO
# ---------------------------------------------------

difference = abs(
    analytic_grad - numeric_grad
)

print("\n=== GRADIENT CHECK ===\n")

print(f"Analytic Gradient : {analytic_grad}")

print(f"Numeric Gradient  : {numeric_grad}")

print(f"Difference        : {difference}")

# criterio
if difference < 1e-4:
    print("\n✅ Backpropagation correcto")
else:
    print("\n❌ Revisar backward")