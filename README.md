# Evaluación Parcial N°1 – Fundamentos de Deep Learning

## Clasificación de imágenes CIFAR-10 con Perceptrón Multicapa (MLP)

**Curso:** DLY0100 – Deep Learning
**Integrantes:** Esteban Pacheco Vega y Francisco Figueroa Donoso
**Fecha:** 21-09-2026

---

## Descripción

Este proyecto implementa, entrena y evalúa una red neuronal **feed-forward** (Perceptrón Multicapa, MLP) —sin capas convolucionales— capaz de clasificar imágenes del dataset **CIFAR-10** (Krizhevsky, 2009). Cada imagen (32×32×3 píxeles RGB) se aplana en un vector de 3.072 valores que alimenta la red, cuya salida es una de las 10 clases del dataset: avión, auto, pájaro, gato, ciervo, perro, rana, caballo, barco y camión.

El desarrollo se realiza en **TensorFlow/Keras** y las visualizaciones en **Plotly**. Todo el trabajo está contenido en un único notebook, organizado como un experimento metodológico reproducible.

### Objetivos

- Cargar y preprocesar adecuadamente los datos de imagen.
- Configurar y analizar hiperparámetros clave (épocas, tasa de aprendizaje, tamaño de batch, capacidad de la red).
- Comparar funciones de activación y funciones de error (loss).
- Aplicar técnicas de optimización y regularización (Dropout, L2, Early Stopping).
- Evaluar el modelo con métricas estándar de clasificación (accuracy, precision, recall, F1-score).
- Analizar críticamente los resultados y justificar las decisiones tomadas.

### Nota metodológica

Entrenar el MLP sobre las 50.000 imágenes de entrenamiento para **cada** experimento resulta poco práctico en un entorno sin GPU. Por ello se trabaja con un **subconjunto reducido y estratificado** del dataset:

- **Entrenamiento:** 9.000 imágenes (900 por clase).
- **Validación:** 4.000 imágenes (400 por clase) — monitoreo y Early Stopping.
- **Test:** 2.000 imágenes (200 por clase) — reservado exclusivamente para la evaluación final.

Además, un MLP no explota la estructura espacial de la imagen (a diferencia de una CNN), por lo que su techo de desempeño en CIFAR-10 es intrínsecamente limitado. Esto se retoma en las conclusiones.

---

## Requisitos y dependencias

- **Python 3.10+** (o Google Colab, que ya incluye la mayoría de las librerías).
- Las siguientes librerías de Python:

| Librería        | Uso |
|-----------------|-----|
| `tensorflow`    | Construcción, entrenamiento y regularización del MLP; carga de CIFAR-10 (`keras.datasets.cifar10`). |
| `keras`         | API de alto nivel para definir el modelo y los callbacks (incluida con TensorFlow). |
| `numpy`         | Manejo de arreglos numéricos (imágenes, semillas, predicciones). |
| `pandas`        | Tablas resumen de experimentos y métricas. |
| `scikit-learn`  | Partición de datos (`train_test_split`) y métricas (`accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`, `classification_report`). |
| `plotly`        | Todas las visualizaciones interactivas del notebook. |
| `jupyter`       | Ejecución local del notebook (incluye `notebook`/`jupyterlab`). |

Instalación de las dependencias:

```bash
pip install tensorflow numpy pandas scikit-learn plotly jupyter
```

> **Dataset:** no es necesario descargar CIFAR-10 manualmente. El notebook llama a
> `keras.datasets.cifar10.load_data()`, que descarga automáticamente los datos (~170 MB)
> en `~/.keras/datasets/` la primera vez que se ejecuta.

---

## Instrucciones de ejecución

### Opción A — Google Colab (recomendado)

1. Abrir el archivo `EP1_DLY0100_MLP_CIFAR10.ipynb` en Google Colab.
2. Seleccionar `Entorno de ejecución → Ejecutar todo`.
3. Aceptar la advertencia de ejecución: el notebook entrena **múltiples** modelos.

> Sin GPU, el entrenamiento completo puede tardar bastante. Si se necesita reducir el tiempo,
> disminuir los valores de `N_TRAIN` / `N_VAL` / `N_TEST` o de las constantes de épocas
> (`EPOCHS_BASE`, `EPOCHS_EXPERIMENTO`, `EPOCHS_REG`).

### Opción B — Entorno local

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/Fran-informatica/Deep-Learning-Eva-1.git
   cd Deep-Learning-Eva-1
   ```

2. (Opcional pero recomendado) Crear y activar un entorno virtual:

   ```bash
   python -m venv .venv
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. Instalar las dependencias:

   ```bash
   pip install tensorflow numpy pandas scikit-learn plotly jupyter
   ```

4. Iniciar Jupyter y abrir el notebook:

   ```bash
   jupyter notebook
   ```

5. Ejecutar las celdas **en orden**, de arriba hacia abajo.

### Reproducibilidad

El notebook fija la semilla `SEED = 42` en NumPy y TensorFlow, de modo que los pesos iniciales, el muestreo del subconjunto de datos y el orden de los batches sean —en la medida de lo posible— reproducibles entre ejecuciones. Aun así, ciertas operaciones en GPU pueden introducir pequeñas diferencias no deterministas.

---

## Estructura del proyecto

```
Deep-Learning-Eva-1/
├── EP1_DLY0100_MLP_CIFAR10.ipynb   # Notebook principal (desarrollo completo)
└── README.md                        # Este archivo
```

### Secciones del notebook

| N° | Sección |
|----|---------|
| 1  | Introducción (descripción del problema, objetivo, estructura) |
| 2  | Configuración inicial e importación de librerías |
| 3  | Carga y exploración de los datos (visualización y distribución de clases) |
| 4  | Preprocesamiento (muestreo estratificado y normalización de píxeles) |
| 5  | Utilidades de visualización (Plotly) |
| 6  | Definición de la arquitectura del modelo (MLP) |
| 7  | Entrenamiento del modelo base y curvas de aprendizaje |
| 8  | Análisis del efecto de los hiperparámetros (learning rate, batch size, capacidad) |
| 9  | Comparación de funciones de activación |
| 10 | Comparación de funciones de error (loss) |
| 11 | Técnicas de optimización y regularización (Dropout, L2, Early Stopping) |
| 12 | Ajuste final de hiperparámetros y reentrenamiento |
| 13 | Evaluación del modelo en el conjunto de test (métricas y matriz de confusión) |
| 14 | Comparación de configuraciones — tabla resumen general |
| 15 | Conclusiones generales |
| 16 | Referencias |
| 17 | Anexo — Herramientas y tecnologías utilizadas |

---

## Herramientas y tecnologías

- **Entorno de ejecución:** Google Colab.
- **Frameworks y librerías:** TensorFlow, Keras, scikit-learn, NumPy, Pandas y Plotly.
- **Documentación y apoyo:** Google Docs y asistentes de IA (Claude, Gemini) para estructuración y redacción.

---

## Referencias

- Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical report, University of Toronto. [https://cave.cs.toronto.edu/kriz/cifar.html](https://cave.cs.toronto.edu/kriz/cifar.html)
