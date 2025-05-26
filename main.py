import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

# Cargar el dataset de California Housing
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

print("Primeras filas del dataset:")
print(X.head())

print("\nVariable objetivo (precio de vivienda):")
print(y[:5])

# -------------------------------
# 1. Regresión Lineal Simple
# Usamos solo la columna 'MedInc' (Ingreso Medio)
# -------------------------------

X_simple = X[['MedInc']].values  # Convertimos a array de NumPy

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

# Normalizamos los datos (importante para redes neuronales y regresión)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creamos el modelo de regresión lineal con Keras
model = models.Sequential([
    layers.Dense(1, input_shape=(1,))
])

# Compilamos el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenamos el modelo
history = model.fit(X_train_scaled, y_train, epochs=50, verbose=0)

# Hacemos predicciones
y_pred = model.predict(X_test_scaled)

# Graficamos resultados: valores reales vs predicciones
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Valores Reales', alpha=0.7)
plt.scatter(X_test, y_pred, label='Predicciones', color='red', alpha=0.7)
plt.title('Regresión Lineal Simple - Precio de Vivienda vs Ingreso Medio')
plt.xlabel('Ingreso Medio (MedInc)')
plt.ylabel('Precio de la Vivienda')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------
# 2. Red Neuronal con múltiples variables
# Usamos todas las características del dataset
# -------------------------------

X_multi = X.values  # Usamos todas las columnas

# Dividimos los datos
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)

# Normalización para múltiples características
scaler_multi = StandardScaler()
X_train_scaled_m = scaler_multi.fit_transform(X_train_m)
X_test_scaled_m = scaler_multi.transform(X_test_m)

# Construimos la red neuronal
model_nn = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled_m.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Salida: precio de la vivienda
])

# Compilamos el modelo
model_nn.compile(optimizer='adam', loss='mse')

# Entrenamos la red neuronal
history_nn = model_nn.fit(
    X_train_scaled_m, y_train_m,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluamos el modelo
loss = model_nn.evaluate(X_test_scaled_m, y_test_m)
print(f"\nError cuadrático medio en prueba (Red Neuronal): {loss:.4f}")

# Hacemos predicciones
y_pred_nn = model_nn.predict(X_test_scaled_m)

# Graficamos comparación entre valores reales y predicciones
plt.figure(figsize=(12, 5))
plt.plot(y_test_m[:50], label='Valores Reales', marker='o', linestyle='')
plt.plot(y_pred_nn[:50], label='Predicciones', marker='x', linestyle='', alpha=0.8)
plt.title('Comparación: Valores Reales vs Predicciones (Red Neuronal)')
plt.xlabel('Índice de muestra')
plt.ylabel('Precio de la vivienda')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar mensaje sobre desempeño del modelo
if loss < 0.5:
    print("✅ El modelo tiene buen desempeño.")
else:
    print("⚠️ El modelo necesita ajustes.")

# Guardar gráfico final (opcional)
plt.savefig('resultados_regresion.png')