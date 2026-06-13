from ocr import extract_text
from features import code_to_features
from mlp import MLP
import numpy as np

# clases
classes = ["O(1)", "O(log n)", "O(n)", "O(n²)"]

# modelo
model = MLP(10, 8, 4)

# leer imagen
text = extract_text("codigo.png")

print("\n=== TEXTO DETECTADO ===")
print(text)

# convertir a features
X = code_to_features(text)

# predecir
pred = model.forward(X)

label = classes[np.argmax(pred)]

print("\n=== PREDICCIÓN ===")
print(label)