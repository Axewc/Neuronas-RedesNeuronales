# Redes Neuronales Artificiales para el Análisis de Curvas de Bragg

## Introducción

Este proyecto aplica Redes Neuronales Artificiales (RNA) al análisis de Curvas de Bragg. Las Curvas de Bragg describen la pérdida de energía de la radiación ionizante al atravesar la materia. El objetivo principal es estimar dos parámetros clave de estas curvas: la Energía Total de un ion (E) y el Pico de la Curva (PB). [cite: 156, 157, 158, 159, 160, 176, 177, 183, 184]

El análisis digital de forma de pulsos (DPSA) con redes neuronales artificiales permite la estimación de estos parámetros. [cite: 177] El proyecto se enmarca dentro del aprendizaje supervisado, donde se entrena una RNA para predecir E y PB a partir de un conjunto de datos de entrada. [cite: 158, 159, 160, 177]

## Metodología

### 1.  Datos

* **Fuente de datos:** Las curvas de Bragg y sus variables descriptivas se obtuvieron a partir del Análisis Digital de Señales. [cite: 161, 162]
* **Características del dataset:**
  * El dataset utilizado contiene datos sintéticos de Curvas de Bragg con un 10% de ruido. [cite: 158, 159, 160, 161]
  * Cada curva se compone de 81 valores descriptivos. [cite: 159, 160]
  * El dataset incluye 100 muestras de curvas de Bragg, con 451 clases por curva. [cite: 160, 162]
  * Las variables de salida (objetivo) son la Energía (E) y el Pico de Bragg (PB). [cite: 159, 160]
  * El dataset se encuentra estandarizado, con valores en el intervalo de 0 a 1. [cite: 161]
* **Preprocesamiento de datos:**
  * Los datos numéricos experimentales se consideran esenciales y no se eliminan atributos ni se añaden ceros. [cite: 174, 175]
  * No se realiza escalado de atributos ya que los valores objetivo ya están en el rango de 0-1. [cite: 179, 180]
  * Todos los atributos se consideran relevantes. [cite: 181, 182]

### 2.  Modelo de Red Neuronal

* **Tipo de red:** Se utiliza un Perceptrón Multicapa (MLP). [cite: 159, 160, 184]
* **Arquitectura:**
  * Capa de entrada: 81 unidades. [cite: 188]
  * Capas ocultas: Tres capas con 33, 39 y 9 unidades, respectivamente. [cite: 188]
  * Capa de salida: 2 unidades (para predecir Energía y Pico de Bragg). [cite: 188, 189]
* **Función de activación:** Sigmoide. [cite: 191, 212, 213]
* **Función de pérdida:** Error Absoluto Medio. [cite: 191]
* **Optimizador:** Adam. [cite: 191]
* **Métricas:** Error Cuadrático Medio. [cite: 191]

### 3.  Entrenamiento del Modelo

* **Búsqueda de hiperparámetros:** Se utilizó la técnica Hyperband para la optimización de hiperparámetros. [cite: 190, 191]
* **Early Stopping:** Se implementó Early Stopping para prevenir el sobreajuste, monitoreando la pérdida de validación. [cite: 191]
* **Validación Cruzada K-Fold:** Se utilizó la validación cruzada K-Fold (k=10) para evaluar la generalización del modelo. [cite: 196, 197, 198]
* **División de datos:** El conjunto de datos se dividió en conjuntos de entrenamiento y prueba. [cite: 185, 186]

### 4.  Implementación en Python

El modelo se implementó utilizando las bibliotecas de Python `tensorflow` y `keras-tuner`. A continuación, se describen brevemente las etapas clave del código:

* **Carga de datos:** Se cargan los datos desde un archivo CSV utilizando `pandas`. [cite: 164]
* **Preprocesamiento:** Se separan las características (X) y los objetivos (Y). [cite: 174, 175, 179, 180, 181, 182]
* **Definición del modelo:** Se define la arquitectura del MLP utilizando `tensorflow.keras.models.Sequential`. [cite: 188, 189]
* **Optimización de hiperparámetros:** Se utiliza `keras_tuner.tuners.Hyperband` para encontrar los mejores hiperparámetros.
* **Entrenamiento del modelo:** El modelo se entrena con los datos de entrenamiento utilizando `model.fit`. [cite: 190, 191]
* **Evaluación del modelo:** Se evalúa el rendimiento del modelo en el conjunto de prueba utilizando `model.evaluate`.
* **Visualización de resultados:** Se utilizan `matplotlib` y `seaborn` para visualizar los resultados, incluyendo la pérdida de validación y las predicciones del modelo.

### 5.  Resultados

El modelo entrenado se evaluó en un conjunto de prueba. Los resultados de la evaluación se presentan a continuación:

* Test loss: 0.0018360227113589644
* Test accuracy: 1.8790286048897542e-05 [cite: 193, 288]

Además, se analizaron las curvas de pérdida de validación y la precisión del modelo durante el entrenamiento. [cite: 195, 196, 289, 290, 291, 292, 293, 294]

### 6.  Discusión

El proyecto demostró la aplicabilidad de las Redes Neuronales Artificiales para la estimación de parámetros en Curvas de Bragg. La metodología implementada, que incluye la optimización de hiperparámetros y la validación cruzada, contribuyó a la robustez del modelo.

Los desafíos encontrados durante el desarrollo incluyeron la búsqueda de hiperparámetros óptimos y las limitaciones de los recursos computacionales para entrenar modelos con una gran cantidad de épocas. [cite: 198, 199, 200, 201]

### 7.  Conclusiones

Este proyecto presenta un modelo de red neuronal para analizar las curvas de Bragg, capaz de predecir la energía y el pico de la curva. El uso de técnicas como la optimización de hiperparámetros y la validación cruzada permitió obtener resultados satisfactorios. [cite: 193, 288]

Trabajos futuros podrían enfocarse en la exploración de diferentes arquitecturas de redes neuronales, la incorporación de un mayor número de variables predictoras y un análisis más profundo del impacto del ruido en el rendimiento del modelo. [cite: 200, 201]

### 8.  Bibliografía

1. Learning limits of an artificial neural network, por J.J. Vega y R. Reynosa, y H. Carrillo Calvet. [cite: 202, 302]
2. Redes Neuronales Artificiales para el Reconocimiento de Curvas de Bragg, por C. RENDÓN BARRAZA. [cite: 203, 303]
3. Regularization methods vs large training sets, por J.J. Vega, H. Carrillo Calvet y José Luis Jiménez Andrade. [cite: 204, 304]
