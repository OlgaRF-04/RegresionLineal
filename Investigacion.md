# 🧠 Investigación Teórica: Fundamentos de Redes Neuronales y Regresión Lineal

## ¿Qué es una red neuronal artificial?

Una **red neuronal artificial (RNA)** es un modelo computacional inspirado en el funcionamiento del cerebro humano. Está compuesta por **neuronas artificiales interconectadas** que procesan información mediante capas: **entrada, ocultas y salida**. Cada conexión tiene un peso ajustable que permite al modelo aprender patrones a partir de datos.

Las redes neuronales son fundamentales en aprendizaje profundo (**deep learning**) y se utilizan para resolver problemas complejos como clasificación, predicción y reconocimiento de patrones.

### Componentes básicos de una RNA:
- **Capa de entrada**: Recibe los datos.
- **Capas ocultas**: Procesan la información con funciones no lineales.
- **Capa de salida**: Proporciona el resultado final.
- **Función de activación**: Introduce no linealidad (ej.: ReLU, Sigmoid, Tanh).
- **Pesos y sesgos**: Parámetros que se ajustan durante el entrenamiento.

---

## ¿Qué es una regresión lineal y cómo se aplica en ML?

La **regresión lineal** es una técnica estadística que modela la relación entre una variable dependiente (objetivo) y una o más variables independientes (características), asumiendo que esta relación es lineal:

$$ y = m \cdot x + b $$

En **Machine Learning**, se usa para predecir valores numéricos, como precios o temperaturas. Por ejemplo, puede usarse para estimar el precio de una vivienda basándose en el tamaño de la casa.

### Tipos de regresión lineal:
- **Regresión lineal simple**: Usa solo 1 característica (ej.: `MedInc`).
- **Regresión lineal múltiple**: Usa varias características (ej.: `AveRooms`, `HouseAge`, etc.).

Es uno de los modelos más simples pero muy útil como punto de partida en proyectos de predicción.

---

## ¿Para qué sirve TensorFlow/Keras en el desarrollo de modelos ML?

**TensorFlow** es una biblioteca de código abierto desarrollada por Google para construir y entrenar modelos de Machine Learning e Inteligencia Artificial. **Keras** es una API de alto nivel que se ejecuta sobre TensorFlow (entre otras plataformas), facilitando la creación de modelos de redes neuronales con una sintaxis clara y modular.

### Funcionalidades clave:
- Definir arquitecturas de redes neuronales.
- Entrenar modelos con grandes volúmenes de datos.
- Optimizar parámetros y evaluar su rendimiento.
- Implementar soluciones escalables en áreas como visión artificial, procesamiento de lenguaje natural y series temporales.

Estas herramientas permiten crear desde modelos sencillos hasta redes profundas complejas con pocas líneas de código.

---

## ¿Por qué es importante el control de versiones con GitHub?

GitHub es una plataforma basada en **Git**, un sistema de control de versiones que permite:

- Guardar cambios progresivos del código.
- Trabajar en equipo sin sobreescribir archivos.
- Revisar historial de modificaciones.
- Publicar proyectos y recibir retroalimentación.
- Mantener una estructura organizada y colaborativa.

### Ventajas en proyectos de Machine Learning:
- Permite mantener un historial claro del desarrollo.
- Facilita la integración continua y pruebas automatizadas.
- Ayuda a compartir resultados con compañeros o instructores.
- Es una práctica profesional estándar en la industria.

---
