# 🏠 Predicción de Precios de Viviendas usando Redes Neuronales

Este proyecto tiene como objetivo construir y evaluar modelos de Machine Learning para predecir los precios de viviendas utilizando el dataset **California Housing**. Se implementa primero un modelo de **regresión lineal simple** y luego una **red neuronal artificial** usando **TensorFlow y Keras**.

---

## 🧠 Objetivo

- Aprender los fundamentos de regresión lineal y redes neuronales.
- Aplicar técnicas de procesamiento de datos y visualización.
- Entrenar y comparar modelos predictivos con Python.
- Usar GitHub para control de versiones y documentación del proyecto.

---

## 📦 Tecnologías Utilizadas

- **Python 3.12**
- **TensorFlow / Keras**: Para construcción y entrenamiento de la red neuronal.
- **Scikit-learn**: Carga del dataset y preparación de datos.
- **Pandas & NumPy**: Manipulación de datos.
- **Matplotlib**: Visualización de resultados.
- **StandardScaler**: Normalización de características.

---

## 📁 Estructura del Proyecto

RedNeuronal_PreciosViviendas/
├── main.py # Código principal del proyecto
├── README.md # Este archivo


---

## ▶️ Cómo Ejecutar el Proyecto

### 1. Clona el repositorio:

git clone https://github.com/tuusuario/RedNeuronal_PreciosViviendas.git 
cd RedNeuronal_PreciosViviendas

### 2. Crea e inicializa el entorno virtual (opcional pero recomendado)

python -m venv venv
venv\Scripts\activate

### 3. Instala las dependencias:

pip install numpy pandas matplotlib scikit-learn tensorflow jupyter

### 4. Ejecuta el script: 

python main.py

## 📊 Resultados Obtenidos

El script genera dos modelos:

### 1. Regresión Lineal Simple

Usa solo la característica MedInc (ingreso medio en el bloque).
Muestra gráficamente las predicciones vs valores reales.

### 2. Red Neuronal Artificial

Arquitectura: 2 capas ocultas con 64 neuronas ReLU.
Pérdida: MSE (Mean Squared Error)
Evaluación: Se imprime el error cuadrático medio final.
También se muestra un gráfico comparativo de predicciones vs valores reales.
