# Eco-asistente: cómo funciona la aplicación

Este documento explica el flujo de `app.py`, revisa si el clasificador aplica PLN y presenta su evaluación por categoría.

## Resumen del proyecto

La aplicación recibe una descripción escrita de un residuo y estima una de seis etiquetas que existen actualmente en `datos/dataset_eco_asistente.csv`:

| Etiqueta del dataset | Interpretación que muestra la app |
|---|---|
| `amarillo` | Contenedor amarillo: envases ligeros |
| `azul` | Contenedor azul: papel y cartón |
| `gris` | Contenedor gris: resto |
| `marron` | Contenedor marrón: biorresiduos |
| `verde` | Contenedor verde: vidrio |
| `punto_limpio` | Punto limpio; es un destino de recogida, no un contenedor de acera |

La aplicación usa estas etiquetas porque el CSV actual está organizado así. El esquema de recogida y las instrucciones concretas varían entre municipios españoles; el resultado es una orientación general.

## Recorrido del código, por secciones

### 1. Bibliotecas y ubicación del CSV

```python
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datos" / "dataset_eco_asistente.csv"
```

- `Path` construye rutas de archivos sin depender del sistema operativo.
- `pandas` carga y prepara la tabla CSV.
- `streamlit` crea la interfaz web.
- `CountVectorizer` convierte frases en conteos numéricos de palabras.
- `LogisticRegression` aprende a asignar esos conteos a una categoría.
- Las funciones de `sklearn.metrics` y `train_test_split` permiten evaluar el modelo con ejemplos reservados.
- `make_pipeline` une la conversión de texto y el clasificador en un solo objeto.
- `__file__` indica dónde está `app.py`; así se encuentra el CSV aunque se inicie Streamlit desde otra carpeta.

### 2. Etiquetas y consejos

Los diccionarios `NOMBRES_CONTENEDOR` y `CONSEJOS` traducen las etiquetas del CSV a texto comprensible. La clave debe coincidir exactamente con la clase que devuelve el modelo. `punto_limpio` tiene un mensaje aparte porque no es uno de los cinco contenedores habituales.

### 3. Cargar, limpiar, dividir y entrenar

La función `cargar_recursos()` concentra el flujo de datos y modelos. El decorador `@st.cache_resource` conserva el modelo mientras la app está activa para no entrenarlo de nuevo en cada interacción.

```python
datos = pd.read_csv(DATASET_PATH)
datos = datos.dropna(subset=["residuo", "categoria"]).copy()
datos["residuo"] = datos["residuo"].astype(str).str.strip().str.lower()
datos["categoria"] = datos["categoria"].astype(str).str.strip().str.lower()
```

Se carga el CSV, se eliminan las filas sin descripción o categoría, y se normalizan espacios y mayúsculas. Después, `residuo` se usa como entrada (`X`) y `categoria` como respuesta correcta (`y`).

```python
train_test_split(
    textos,
    etiquetas,
    test_size=0.25,
    random_state=42,
    stratify=etiquetas,
)
```

Se reservan 25 % de los ejemplos para la prueba. `stratify` intenta conservar la proporción de cada categoría en ambos grupos y `random_state=42` hace reproducible la partición. El modelo no aprende de los ejemplos de prueba durante esta etapa.

```python
modelo_evaluacion = make_pipeline(
    CountVectorizer(),
    LogisticRegression(max_iter=1000, random_state=42),
)
modelo_evaluacion.fit(textos_entrenamiento, etiquetas_entrenamiento)
predicciones = modelo_evaluacion.predict(textos_prueba)
```

`fit` aprende primero el vocabulario y los conteos de palabras y luego ajusta el clasificador. `predict` produce etiquetas para los textos reservados. Al final se crea otro pipeline, `modelo_final`, y se entrena con todos los ejemplos disponibles para que la app use la mayor cantidad de datos al responder.

### 4. Interfaz de Streamlit

- `st.set_page_config` define el título y el icono de la pestaña.
- `st.title`, `st.write` y `st.caption` presentan la aplicación e indican ejemplos de consulta.
- `st.text_input` crea la caja para escribir el residuo.
- `st.button("Consultar")` ejecuta la predicción cuando se pulsa el botón.
- `residuo.strip().lower()` quita espacios exteriores y pasa el texto a minúsculas.
- `modelo.predict([...])[0]` pide una predicción. Se usa una lista porque Scikit-learn espera una colección de textos, aunque sea una sola consulta.
- `NOMBRES_CONTENEDOR.get(...)` muestra el nombre legible asociado a la clase; `CONSEJOS.get(...)` añade el consejo.
- `st.expander` agrupa los gráficos y métricas para que se puedan abrir durante la demostración.
- El botón `Salir` borra el estado de esa sesión del navegador. No apaga el servidor; para eso se pulsa Ctrl+C en la terminal.

### 5. Qué representan las gráficas

La app enseña el número de filas disponibles por clase y compara aciertos con errores por categoría en el conjunto de prueba. La matriz de confusión utiliza filas para la categoría real y columnas para la predicha. Los valores de la diagonal son aciertos; los valores fuera de la diagonal son confusiones.

## ¿Es realmente PLN y aprendizaje automático?

**Sí, en el sentido de clasificación de texto con PLN tradicional.** `CountVectorizer` convierte cada descripción en una representación de conteos de palabras, y `LogisticRegression` aprende, usando ejemplos etiquetados, qué clases se relacionan con esos conteos. Scikit-learn describe `CountVectorizer` como un conversor de documentos a una matriz de conteos de tokens.

El alcance es sencillo: con la configuración por defecto utiliza palabras individuales (unigramas). No comprende el significado como una persona ni reconoce materiales que no estén bien representados en el dataset. Por ejemplo, puede asociar `plástico` con envases aunque el objeto de plástico no sea un envase. No hay validación de confianza ni detector de consultas fuera del vocabulario; por tanto, siempre devuelve alguna categoría.

El método es correcto como ejercicio introductorio de clasificación de texto, pero los resultados actuales no son suficientemente equilibrados para afirmar que clasifica bien cualquier residuo. La calidad de las etiquetas y la cobertura de ejemplos afectan directamente al modelo.

## Evaluación reproducible del CSV actual

La matriz y las métricas siguientes se obtuvieron con los mismos parámetros del notebook y la app: `test_size=0.25`, `random_state=42`, partición estratificada, `CountVectorizer` y `LogisticRegression(max_iter=1000)`. En el CSV usado para este cálculo hay **160 filas**: amarillo 19, azul 8, gris 46, marrón 10, punto limpio 75 y verde 2. El conjunto de prueba contiene 40 ejemplos.

### Matriz de confusión

Filas = categoría real. Columnas = predicción del modelo. Orden de las columnas y filas: amarillo, azul, gris, marrón, punto limpio, verde.

| Real ↓ / Predicha → | Amarillo | Azul | Gris | Marrón | Punto limpio | Verde |
|---|---:|---:|---:|---:|---:|---:|
| **Amarillo** | 3 | 0 | 1 | 0 | 1 | 0 |
| **Azul** | 0 | 0 | 1 | 0 | 1 | 0 |
| **Gris** | 0 | 0 | 4 | 0 | 7 | 0 |
| **Marrón** | 0 | 0 | 1 | 0 | 1 | 0 |
| **Punto limpio** | 1 | 0 | 2 | 0 | 16 | 0 |
| **Verde** | 0 | 0 | 0 | 0 | 1 | 0 |

La exactitud total es **57,5 % (23 de 40)**. El modelo predice con mucha frecuencia `punto_limpio`, que es la clase mayoritaria del dataset; por eso la exactitud por sí sola puede dar una impresión demasiado favorable. No acertó ejemplos de azul, marrón ni verde en esta partición de prueba.

### Métricas por clase

| Categoría | Ejemplos de prueba | Precisión | Recall | F1 |
|---|---:|---:|---:|---:|
| Amarillo | 5 | 0,75 | 0,60 | 0,67 |
| Azul | 2 | 0,00 | 0,00 | 0,00 |
| Gris | 11 | 0,44 | 0,36 | 0,40 |
| Marrón | 2 | 0,00 | 0,00 | 0,00 |
| Punto limpio | 19 | 0,59 | 0,84 | 0,70 |
| Verde | 1 | 0,00 | 0,00 | 0,00 |

La precisión responde a «cuando predice esta clase, ¿qué proporción de esas predicciones es correcta?». El recall responde a «de los residuos que realmente pertenecen a esta clase, ¿cuántos encuentra?». F1 combina ambas medidas. En las clases con uno o dos ejemplos de prueba, estas métricas son muy inestables; la clase verde tiene solo dos filas en todo el CSV.

### Conclusión para la presentación

> «Usamos PLN clásico: CountVectorizer representa el texto mediante conteos de palabras y LogisticRegression aprende a clasificarlo. Evaluamos con ejemplos reservados. La exactitud fue 57,5 %, pero el dataset está desequilibrado: punto limpio domina y azul, marrón y verde tienen muy pocos ejemplos. El modelo sirve para demostrar el proceso, no para dar recomendaciones fiables en toda España».

## Próximas mejoras recomendadas

1. Reequilibrar el dataset incorporando más ejemplos revisados para azul, marrón y verde.
2. Revisar si cada etiqueta describe un contenedor ordinario o un destino especial; en particular, `punto_limpio` no es un contenedor.
3. Separar la evaluación antes de ajustar el modelo y evitar variantes casi duplicadas entre entrenamiento y prueba.
4. Probar `ngram_range=(1, 2)` para que el vectorizador pueda usar pares de palabras además de términos sueltos.
5. Añadir manejo de consultas desconocidas o de baja confianza y revisar el efecto antes de presentar la app como orientador general.

## Referencias técnicas

- [Documentación de scikit-learn: CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Documentación de scikit-learn: matriz de confusión](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [MITECO: modelos de gestión y fracciones municipales](https://www.miteco.gob.es/es/calidad-y-evaluacion-ambiental/temas/prevencion-y-gestion-residuos/flujos/domesticos/gestion/modelo_gestion.html)
