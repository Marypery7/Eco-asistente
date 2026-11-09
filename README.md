# Eco-asistente

Aplicación educativa que clasifica residuos escritos en texto y ofrece una orientación sobre la categoría de reciclaje estimada. El proyecto utiliza Python, pandas, `CountVectorizer`, Scikit-learn y Streamlit.

> La clasificación es orientativa. El dataset es pequeño y las normas de recogida pueden variar entre municipios de España. Consulta las instrucciones locales para decidir dónde depositar un residuo.

## Contenido del proyecto

```text
.
├── app.py                                # Aplicación web Streamlit
├── requirements.txt                      # Bibliotecas necesarias
├── datos/
│   └── dataset_eco_asistente.csv          # Ejemplos de residuos etiquetados
└── notebooks/
    └── eco_asistente_analisis.ipynb      # Análisis, preparación y evaluación
```

El dataset contiene ejemplos elaborados para este proyecto, con las categorías **papel y cartón**, **vidrio**, **envases**, **orgánico** y **resto**. El notebook permite revisar las categorías, entrenar y evaluar el clasificador y analizar errores.

## Requisitos

- Python 3.10 o posterior.
- Conexión a Internet durante la instalación de las dependencias.
- Navegador web para utilizar la aplicación.

## Descargar el proyecto desde GitHub

Puedes descargar el repositorio como ZIP desde el botón **Code → Download ZIP** de GitHub y descomprimirlo. También puedes clonarlo desde una terminal:

```bash
git clone https://github.com/patomayi/proyecto-final-impulsia100h.git
cd proyecto-final-impulsia100h
```

El repositorio del proyecto es [patomayi/proyecto-final-impulsia100h](https://github.com/patomayi/proyecto-final-impulsia100h).

## Instalar y ejecutar

Crea un entorno virtual dentro de la carpeta del proyecto. Así las bibliotecas se instalan para este proyecto y no en el Python del sistema.

### Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS y Linux (Bash o Zsh)

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

### Linux (Fish)

Los comandos directos al Python del entorno virtual también funcionan en Fish:

```fish
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Si prefieres activar el entorno en Fish antes de usar `python`, ejecuta `source .venv/bin/activate.fish`. En Bash o Zsh, el activador es `source .venv/bin/activate`; en PowerShell, ` .venv\Scripts\Activate.ps1`.

Al iniciar, Streamlit mostrará una dirección local en la terminal y normalmente abrirá la app en el navegador. Para detener el servidor, vuelve a esa terminal y pulsa **Ctrl+C**.

## Uso de la aplicación

1. Escribe un residuo en el campo de texto, incluyendo el material cuando sea necesario; por ejemplo, `botella de plástico` o `frasco de vidrio`.
2. Pulsa **Consultar**.
3. La aplicación muestra la categoría estimada y un consejo breve.
4. Abre **Ver análisis de datos y evaluación del modelo** para consultar gráficos del dataset y de los aciertos y errores del clasificador.

El botón **Salir** cierra la sesión de la página y borra su estado. Para apagar el servidor local, pulsa **Ctrl+C** en la terminal.

## Abrir el notebook en Google Colab

1. Sube `notebooks/eco_asistente_analisis.ipynb` a Google Colab.
2. Adjunta `datos/dataset_eco_asistente.csv` en Colab cuando el notebook lo solicite, o colócalo junto al notebook en el mismo entorno.
3. Ejecuta las celdas en orden.

El notebook también puede abrirse localmente en Jupyter Notebook o JupyterLab. Instala las dependencias del proyecto y cualquier paquete de Jupyter que necesites en el mismo entorno.

## Cómo funciona el clasificador

La app entrena un `Pipeline` de Scikit-learn con dos pasos:

1. `CountVectorizer` transforma las palabras de cada descripción en una representación numérica basada en las palabras que aparecen.
2. `LogisticRegression` aprende a relacionar esas representaciones con las categorías del dataset.

Para mostrar la evaluación, el programa reserva una parte de los ejemplos como conjunto de prueba, mide las predicciones sobre esos ejemplos y muestra la exactitud, un gráfico de aciertos y errores y una matriz de confusión. Después, vuelve a entrenar el modelo de la app con todos los ejemplos disponibles.

## Limitaciones

- El dataset contiene pocos ejemplos y fue preparado con fines educativos; el modelo puede equivocarse con residuos que no aparecen en él.
- La evaluación sobre un conjunto de prueba pequeño es una primera aproximación, no una garantía de rendimiento general.
- Los sistemas de separación y las instrucciones concretas varían entre municipios. La app presenta categorías habituales como orientación general para España.
- Los residuos especiales, peligrosos o voluminosos pueden requerir un punto limpio u otro servicio específico; no deben clasificarse automáticamente como una recomendación municipal definitiva.
