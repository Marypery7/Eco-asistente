from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# Esta ruta apunta al CSV aunque ejecutes la app desde otra carpeta.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datos" / "dataset_eco_asistente.csv"

# Traducimos las etiquetas del modelo a nombres de contenedores fáciles de entender.
NOMBRES_CONTENEDOR = {
    "amarillo": "contenedor amarillo (envases ligeros)",
    "azul": "contenedor azul (papel y cartón)",
    "gris": "contenedor gris (resto)",
    "marron": "contenedor marrón (biorresiduos)",
    "punto_limpio": "punto limpio (residuo especial o específico)",
    "verde": "contenedor verde (vidrio)",
}

CONSEJOS = {
    "amarillo": "Vacía el envase antes de depositarlo; las normas concretas pueden variar localmente.",
    "azul": "Deposítalo limpio y, si es una caja, pliégala.",
    "gris": "Esta fracción es para residuos domésticos no recogidos separadamente.",
    "marron": "Deposita solo biorresiduos admitidos por el servicio local.",
    "punto_limpio": "Consulta el punto limpio de tu municipio para conocer las condiciones de entrega.",
    "verde": "Deposita envases de vidrio; separa tapas y tapones cuando corresponda.",
}


@st.cache_resource
# La caché evita volver a entrenar los modelos en cada clic o actualización de la página.
def cargar_recursos():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No encuentro el dataset: {DATASET_PATH}")

    # pandas lee el CSV y permite limpiar/revisar sus columnas con facilidad.
    datos = pd.read_csv(DATASET_PATH)
    datos = datos.dropna(subset=["residuo", "categoria"]).copy()
    datos["residuo"] = datos["residuo"].astype(str).str.strip().str.lower()
    datos["categoria"] = datos["categoria"].astype(str).str.strip().str.lower()
    datos = datos[(datos["residuo"] != "") & (datos["categoria"] != "")]

    textos = datos["residuo"]
    etiquetas = datos["categoria"]

    # Reservamos parte de los ejemplos; el modelo no los verá mientras aprende.
    textos_entrenamiento, textos_prueba, etiquetas_entrenamiento, etiquetas_prueba = train_test_split(
        textos,
        etiquetas,
        test_size=0.25,
        random_state=42,
        stratify=etiquetas,
    )

    # CountVectorizer cuenta palabras y LogisticRegression aprende su relación con cada etiqueta.
    modelo_evaluacion = make_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    modelo_evaluacion.fit(textos_entrenamiento, etiquetas_entrenamiento)
    predicciones = modelo_evaluacion.predict(textos_prueba)

    # Comparamos predicciones con etiquetas reales para medir el desempeño en datos reservados.
    exactitud = accuracy_score(etiquetas_prueba, predicciones)
    matriz = confusion_matrix(
        etiquetas_prueba,
        predicciones,
        labels=sorted(etiquetas.unique()),
    )
    resultados = pd.DataFrame({
        "Categoría": etiquetas_prueba.to_numpy(),
        "Resultado": [
            "Acierto" if real == predicha else "Error"
            for real, predicha in zip(etiquetas_prueba, predicciones)
        ],
    })

    # Para la app final entrenamos con todos los ejemplos disponibles.
    modelo_final = make_pipeline(
        CountVectorizer(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    modelo_final.fit(textos, etiquetas)

    return datos, modelo_final, exactitud, matriz, sorted(etiquetas.unique()), resultados


st.set_page_config(page_title="Eco-asistente", page_icon="")
st.session_state.setdefault("sesion_cerrada", False)


def cerrar_sesion():
    # Borra los datos de esta sesión del navegador; el servidor sigue activo en la terminal.
    for clave in list(st.session_state.keys()):
        del st.session_state[clave]
    st.session_state["sesion_cerrada"] = True


if st.session_state["sesion_cerrada"]:
    st.title("Sesión cerrada")
    st.info("Tus datos de esta sesión se han borrado. Para apagar la aplicación local, pulsa Ctrl+C en la terminal donde ejecutaste Streamlit.")
    st.stop()

st.title("Eco-asistente")
st.write("Escribe el nombre de un residuo y te indicaré su categoría de reciclaje estimada.")
st.caption("Ejemplos: botella de plástico · frasco de vidrio · cuaderno de papel")

try:
    datos, modelo, exactitud, matriz, categorias, resultados = cargar_recursos()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

residuo = st.text_input("¿Qué residuo quieres clasificar?", placeholder="Ej.: lata de aluminio")

if st.button("Consultar", type="primary"):
    if not residuo.strip():
        st.warning("Escribe un residuo para obtener una predicción.")
    else:
        # El modelo recibe una lista de textos y devuelve la categoría estimada.
        categoria = modelo.predict([residuo.strip().lower()])[0]
        st.subheader("Predicción")
        st.success(NOMBRES_CONTENEDOR.get(categoria, categoria))
        st.write(CONSEJOS.get(categoria, "Consulta las instrucciones de separación locales."))

# Estas gráficas ayudan a explicar tanto los datos como los aciertos y errores del modelo.
with st.expander("Ver análisis de datos y evaluación del modelo"):
    st.subheader("Ejemplos disponibles por categoría")
    conteo_categorias = (
        datos["categoria"]
        .value_counts()
        .reindex(categorias, fill_value=0)
        .rename_axis("Categoría")
        .reset_index(name="Número de ejemplos")
    )
    st.bar_chart(
        conteo_categorias,
        x="Categoría",
        y="Número de ejemplos",
        alt="Cantidad de ejemplos del dataset en cada categoría de residuo",
    )

    st.subheader("Aciertos y errores en los residuos de prueba")
    st.write(f"Exactitud en el conjunto de prueba: **{exactitud:.0%}**")
    resumen = (
        resultados.groupby(["Categoría", "Resultado"])
        .size()
        .unstack(fill_value=0)
        .reindex(categorias, fill_value=0)
        .reset_index()
        .melt(id_vars="Categoría", var_name="Resultado", value_name="Número de residuos")
    )
    st.bar_chart(
        resumen,
        x="Categoría",
        y="Número de residuos",
        color="Resultado",
        stack=False,
        alt="Comparación de aciertos y errores del clasificador en cada categoría",
    )

    st.subheader("Matriz de confusión")
    st.caption("Las filas son las categorías reales; las columnas, las predichas. La diagonal contiene los aciertos.")
    matriz_df = pd.DataFrame(matriz, index=categorias, columns=categorias)
    matriz_df.index.name = "Categoría real"
    st.dataframe(matriz_df, alt="Matriz que muestra aciertos y confusiones por categoría")
    st.caption("La evaluación es inicial: el dataset es pequeño y las normas pueden variar según el municipio.")

st.divider()
st.button("Salir", on_click=cerrar_sesion, help="Cierra esta sesión y borra sus datos.")
st.caption(
    "La predicción es orientativa: el modelo se entrenó con un dataset educativo pequeño. "
    "En España, el sistema de recogida y algunas instrucciones pueden variar según el municipio. "
    "Para residuos especiales o voluminosos, consulta el servicio local o un punto limpio."
)
