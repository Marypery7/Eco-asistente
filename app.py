from pathlib import Path
import re
import unicodedata

import altair as alt
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
    "papel y cartón": "contenedor azul (papel y cartón)",
    "vidrio": "contenedor verde (vidrio)",
    "envases": "contenedor amarillo (envases ligeros)",
    "orgánico": "contenedor marrón (biorresiduos)",
    "resto": "contenedor de resto",
}

COLORES_CONTENEDOR = {
    "papel y cartón": "#4A90D9",
    "vidrio": "#32845F",
    "envases": "#E3B83E",
    "orgánico": "#8A6A4F",
    "resto": "#7D8884",
}

CONSEJOS = {
    "papel y cartón": "Deposítalo limpio y, si es una caja, pliégala.",
    "vidrio": "Vacía el envase y separa tapas o tapones cuando corresponda.",
    "envases": "Vacía el envase antes de depositarlo.",
    "orgánico": "Deposita solo residuos orgánicos admitidos por el servicio local.",
    "resto": "Esta categoría es para residuos domésticos no incluidos en las fracciones selectivas.",
}

UMBRAL_CONFIDENCIA = 0.35
MENSAJE_PUNTO_LIMPIO = "Ir al punto limpio"
MENSAJE_MATERIAL_INDEFINIDO = "Necesito saber el material del envase"

# Listas sencillas para reconocer frutas y vegetales que el modelo de texto
# puede confundir aunque sean ejemplos comunes.
FRUTAS = {
    "manzana", "pera", "platano", "banana", "naranja", "mandarina",
    "limon", "lima", "uva", "fresa", "frutilla", "cereza", "melocoton",
    "durazno", "albaricoque", "damasco", "ciruela", "kiwi", "pina",
    "anana", "mango", "papaya", "melon", "sandia", "aguacate", "palta",
    "granada", "higo", "pomelo", "arandano", "frambuesa", "mora",
    "fruta", "frutas",
}

VEGETALES = {
    "lechuga", "espinaca", "acelga", "col", "repollo", "brocoli",
    "coliflor", "zanahoria", "patata", "papa", "batata", "boniato",
    "camote", "tomate", "pepino", "calabacin", "zucchini", "berenjena",
    "pimiento", "calabaza", "cebolla", "ajo", "puerro", "apio",
    "esparrago", "alcachofa", "judia verde", "judias verdes", "ejote",
    "guisante", "maiz", "maices", "elote", "yuca", "rabano", "remolacha",
    "nabo", "vegetal", "vegetales", "verdura", "verduras",
}

PALABRAS_ENVASE = {
    "botella", "envase", "frasco", "tarro", "lata", "brik", "bote",
    "bandeja", "bolsa", "envoltorio", "plastico", "vidrio", "cristal",
    "metal", "paquete",
}


def normalizar_texto(texto):
    return "".join(
        caracter for caracter in unicodedata.normalize("NFD", str(texto).lower())
        if unicodedata.category(caracter) != "Mn"
    )


def contiene_termino(texto, termino):
    # Acepta el singular y plurales sencillos, con límites de palabra para
    # evitar coincidencias accidentales dentro de otras palabras.
    termino = normalizar_texto(termino)
    if termino.endswith("z"):
        plural = termino[:-1] + "ces"
    elif termino.endswith(("a", "e", "i", "o", "u")):
        plural = termino + "s"
    else:
        plural = termino + "es"
    return any(
        re.search(r"(?<!\w)" + re.escape(variante) + r"(?!\w)", texto)
        for variante in {termino, plural}
    )


def detectar_tipo_fruta_vegetal(residuo):
    texto = normalizar_texto(residuo)
    # Si la descripción habla del envase o del material, se clasifica ese objeto
    # y no el alimento que quizá contenía.
    if any(contiene_termino(texto, palabra) for palabra in PALABRAS_ENVASE):
        return None
    if any(contiene_termino(texto, fruta) for fruta in FRUTAS):
        return "fruta"
    if any(contiene_termino(texto, vegetal) for vegetal in VEGETALES):
        return "vegetal"
    return None
PALABRAS_CLAVE_CATEGORIA = {
    "papel y cartón": [
        "papel", "carton", "cuaderno", "libro", "hoja", "revista", "folleto",
        "sobre", "catalogo", "postal", "caja", "bolsa de papel", "envoltorio de papel",
        "cartulina", "periódico", "diario", "agenda", "cuaderno", "libreta"
    ],
    "vidrio": [
        "vidrio", "frasco", "botella", "tarro", "jarra", "vaso de vidrio",
        "botellin", "botellita", "envase de vidrio", "cristal"
    ],
    "envases": [
        "envase", "plastico", "lata", "aluminio", "brik", "bote", "film", "tapón",
        "aerosol", "bolsa de plastico", "poliestireno", "metal", "tapa", "garrafa"
    ],
    "orgánico": [
        "cascara", "piel", "restos", "posos", "verdura", "fruta", "manzana",
        "naranja", "cafe", "pan", "arroz", "pasta", "cocinado", "queso", "yogur",
        "hojas", "pera", "banana", "ensalada", "brocoli", "tomate", "cebolla", "zanahoria"
    ],
    "resto": [
        "mascarilla", "colilla", "cepillo", "polvo", "bastoncillo", "toallita",
        "esponja", "ceramica", "porcelana", "juguete", "goma", "ropa", "zapato",
        "muñeco", "espejo", "bombilla", "jeringuilla", "dental", "uso", "sucia"
    ],
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


st.set_page_config(page_title="Eco-asistente", page_icon="♻️", layout="centered")
st.session_state.setdefault("sesion_cerrada", False)

st.markdown(
    """
    <style>
    :root {
        --eco-forest: #245b46;
        --eco-leaf: #397b5b;
        --eco-sage: #e7f0e8;
        --eco-cream: #f7f8f3;
        --eco-ink: #1c2d25;
        --eco-muted: #607168;
    }
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(ellipse at 12% 0%, #e8f1e8 0%, #f7f8f3 44%, #f7f8f3 100%);
        color: var(--eco-ink);
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 2.5rem; }
    .eco-hero {
        padding: 1.7rem 1.9rem; margin-bottom: 1.2rem;
        border: 1px solid #d8e5d8; border-radius: 24px;
        background: linear-gradient(120deg, rgba(255,255,255,.96), rgba(235,244,235,.94));
        box-shadow: 0 12px 35px rgba(38, 75, 52, .08);
    }
    .eco-kicker { color: var(--eco-leaf); font-size: .75rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .4rem; }
    .eco-hero h1 { color: var(--eco-forest); font-size: clamp(2rem, 5vw, 3rem); line-height: 1.05; margin: 0 0 .55rem 0; font-weight: 800; }
    .eco-hero p { color: var(--eco-muted); font-size: 1rem; margin: 0; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color: #dce6dd !important; border-radius: 18px !important; background: rgba(255,255,255,.86); box-shadow: 0 7px 22px rgba(38, 75, 52, .045); }
    [data-testid="stTextInputRootElement"] input {
        min-height: 3rem; border-radius: 12px; border-color: #c9d8cb;
        background: #fff !important; color: var(--eco-ink) !important;
        -webkit-text-fill-color: var(--eco-ink) !important; font-size: 1rem;
    }
    [data-testid="stTextInputRootElement"] input::placeholder {
        color: #78877d !important; -webkit-text-fill-color: #78877d !important; opacity: 1;
    }
    [data-testid="stTextInputRootElement"] input:focus { border-color: var(--eco-leaf); box-shadow: 0 0 0 .15rem rgba(57,123,91,.14); }
    [data-testid="stBaseButton-primary"] {
        min-height: 3rem; border: 0 !important; border-radius: 12px !important;
        background: var(--eco-forest) !important; color: #fff !important; font-weight: 700;
    }
    [data-testid="stBaseButton-primary"] p { color: #fff !important; }
    [data-testid="stBaseButton-primary"]:hover { background: var(--eco-leaf) !important; }
    [data-testid="stBaseButton-secondary"] {
        min-height: 2.5rem; border-radius: 12px !important;
        border: 1px solid #c9d8cb !important; background: #edf3ed !important;
        color: var(--eco-forest) !important; font-weight: 700;
    }
    [data-testid="stBaseButton-secondary"] p { color: var(--eco-forest) !important; }
    [data-testid="stBaseButton-secondary"]:hover { background: #dfeae0 !important; border-color: #9eb9a3 !important; }
    [data-testid="stAlert"] { border-radius: 14px; }
    [data-testid="stExpander"] { border: 1px solid #dce6dd; border-radius: 16px; background: rgba(255,255,255,.75); }
    h2, h3 { color: var(--eco-forest); }
    .eco-footer { color: #718077; font-size: .78rem; text-align: center; padding-top: .5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def cerrar_sesion():
    # Borra los datos de esta sesión del navegador; el servidor sigue activo en la terminal.
    for clave in list(st.session_state.keys()):
        del st.session_state[clave]
    st.session_state["sesion_cerrada"] = True


def detectar_categoria_o_punto_limpio(residuo, modelo=None, datos=None):
    # Si no se pasan los recursos, los cargamos juntos. La app pasa ambos
    # explícitamente para reutilizar los recursos ya cargados en su sesión.
    if modelo is None or datos is None:
        datos_cargados, modelo_cargado, *_ = cargar_recursos()
        datos = datos if datos is not None else datos_cargados
        modelo = modelo if modelo is not None else modelo_cargado

    texto = str(residuo or "").strip().lower()
    if not texto:
        return MENSAJE_PUNTO_LIMPIO

    coincidencias = datos[datos["residuo"].str.strip().str.lower() == texto]
    if not coincidencias.empty:
        return coincidencias.iloc[0]["categoria"]

    # Las reglas solo reconocen casos explícitos de residuos especiales que
    # deben salir de las cinco categorías del clasificador.
    texto_normalizado = "".join(
        caracter for caracter in unicodedata.normalize("NFD", texto)
        if unicodedata.category(caracter) != "Mn"
    )
    terminos_especiales = (
        "pila", "bateria", "medicamento", "jeringuilla", "aparato electrico",
        "electrodomestico", "ordenador", "portatil", "laptop", "movil",
        "telefono", "bombilla", "fluorescente", "punto limpio"
    )
    if any(termino in texto_normalizado for termino in terminos_especiales):
        return MENSAJE_PUNTO_LIMPIO

    palabras_envase = (
        "botella", "envase", "frasco", "tarro", "bote", "bandeja",
        "bolsa", "paquete"
    )
    menciona_envase = any(
        contiene_termino(texto_normalizado, palabra) for palabra in palabras_envase
    )
    menciona_vidrio = any(
        contiene_termino(texto_normalizado, palabra) for palabra in ("vidrio", "cristal")
    )
    menciona_ligero = any(
        contiene_termino(texto_normalizado, palabra)
        for palabra in ("plastico", "metal", "aluminio", "brik")
    )
    if menciona_vidrio and menciona_ligero:
        return MENSAJE_MATERIAL_INDEFINIDO
    if menciona_envase and menciona_vidrio:
        return "vidrio"
    if menciona_envase and menciona_ligero:
        return "envases"
    if menciona_envase:
        return MENSAJE_MATERIAL_INDEFINIDO

    # Las latas y los briks ya identifican el tipo de envase ligero aunque no
    # se indique el material en la frase.
    if contiene_termino(texto_normalizado, "lata") or contiene_termino(texto_normalizado, "brik"):
        return "envases"

    # El nombre de una bebida no indica por sí solo si se tira una botella,
    # una lata u otro tipo de envase.
    bebidas = ("cerveza", "vino", "refresco", "agua", "zumo", "leche")
    if any(contiene_termino(texto_normalizado, bebida) for bebida in bebidas):
        return MENSAJE_MATERIAL_INDEFINIDO

    if detectar_tipo_fruta_vegetal(texto):
        return "orgánico"

    # Las palabras clave solo determinan si la entrada pertenece a alguna de
    # las cinco fracciones conocidas; la categoría final siempre la decide el
    # clasificador, evitando que la primera regla coincidente fuerce una etiqueta.
    palabras_conocidas = {
        palabra
        for palabras in PALABRAS_CLAVE_CATEGORIA.values()
        for palabra in palabras
    }
    if not any(
        "".join(caracter for caracter in unicodedata.normalize("NFD", palabra)
                if unicodedata.category(caracter) != "Mn") in texto_normalizado
        for palabra in palabras_conocidas
    ):
        return MENSAJE_PUNTO_LIMPIO

    return modelo.predict([texto])[0]


if st.session_state["sesion_cerrada"]:
    st.title("Sesión cerrada")
    st.info("Tus datos de esta sesión se han borrado. Para apagar la aplicación local, pulsa Ctrl+C en la terminal donde ejecutaste Streamlit.")
    st.stop()

st.markdown(
    """
    <div class="eco-hero">
      <div class="eco-kicker">♻ Eco-asistente</div>
      <h1>Reciclar empieza<br>por identificarlo.</h1>
      <p>Escribe un residuo y recibe una orientación clara sobre dónde depositarlo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    datos, modelo, exactitud, matriz, categorias, resultados = cargar_recursos()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

with st.container(border=True):
    st.subheader("¿Qué quieres tirar?")
    st.caption("Prueba con un nombre y material: botella de plástico, frasco de vidrio o lechuga.")
    residuo = st.text_input(
        "Nombre del residuo",
        placeholder="Ej.: botella de tomate de vidrio",
        label_visibility="collapsed",
    )
    _, boton_col = st.columns([2.2, 1])
    with boton_col:
        consultar = st.button("♻ Clasificar", type="primary", use_container_width=True)

if consultar:
    if not residuo.strip():
        st.warning("Escribe un residuo para obtener una orientación.")
    else:
        with st.container(border=True):
            tipo_detectado = detectar_tipo_fruta_vegetal(residuo)
            categoria = detectar_categoria_o_punto_limpio(residuo, modelo, datos)
            st.markdown("#### Resultado")
            if tipo_detectado:
                st.info(f"Tipo identificado: **{tipo_detectado.capitalize()}**")
            if categoria == MENSAJE_MATERIAL_INDEFINIDO:
                st.warning("Describe qué envase vas a tirar y, si hace falta, su material. Por ejemplo: botella de cerveza de vidrio o lata de cerveza.")
            elif categoria == MENSAJE_PUNTO_LIMPIO:
                st.warning(MENSAJE_PUNTO_LIMPIO)
                st.write("Este residuo no aparece en las categorías del modelo. Prueba con una descripción más concreta.")
            else:
                st.success(f"Deposítalo en: {NOMBRES_CONTENEDOR.get(categoria, categoria)}")
                st.write(CONSEJOS.get(categoria, "Consulta las instrucciones de separación locales."))

# Estas gráficas ayudan a explicar tanto los datos como los aciertos y errores del modelo.
with st.expander("Ver análisis de datos y evaluación del modelo"):
    st.caption("Estos gráficos muestran una evaluación fija sobre ejemplos reservados. No cambian al consultar un residuo nuevo.")
    st.subheader("Ejemplos disponibles por categoría")
    conteo_categorias = (
        datos["categoria"]
        .value_counts()
        .reindex(categorias, fill_value=0)
        .rename_axis("Categoría")
        .reset_index(name="Número de ejemplos")
    )
    grafico_conteo = alt.Chart(conteo_categorias).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6
    ).encode(
        x=alt.X("Categoría:N", sort=categorias, axis=alt.Axis(title=None, labelAngle=-18)),
        y=alt.Y("Número de ejemplos:Q", axis=alt.Axis(title="Ejemplos", gridColor="#E7ECE7")),
        color=alt.Color(
            "Categoría:N",
            scale=alt.Scale(domain=categorias, range=[COLORES_CONTENEDOR[c] for c in categorias]),
            legend=None,
        ),
        tooltip=["Categoría:N", "Número de ejemplos:Q"],
    ).properties(height=250)
    grafico_conteo = grafico_conteo.configure(background="#FFFFFF").configure_view(fill="#FFFFFF", strokeOpacity=0).configure_axis(
        labelColor="#52645A", titleColor="#52645A", tickColor="#DCE5DC"
    )
    st.altair_chart(grafico_conteo, use_container_width=True)

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
    grafico_resultados = alt.Chart(resumen).mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Categoría:N", sort=categorias, axis=alt.Axis(title=None, labelAngle=-18)),
        xOffset=alt.XOffset("Resultado:N", sort=["Acierto", "Error"]),
        y=alt.Y("Número de residuos:Q", axis=alt.Axis(title="Residuos", gridColor="#E7ECE7")),
        color=alt.Color(
            "Categoría:N",
            scale=alt.Scale(domain=categorias, range=[COLORES_CONTENEDOR[c] for c in categorias]),
            legend=None,
        ),
        opacity=alt.Opacity(
            "Resultado:N",
            scale=alt.Scale(domain=["Acierto", "Error"], range=[1, 0.38]),
            legend=alt.Legend(title="Resultado", orient="top"),
        ),
        tooltip=["Categoría:N", "Resultado:N", "Número de residuos:Q"],
    ).properties(height=270)
    grafico_resultados = grafico_resultados.configure(background="#FFFFFF").configure_view(fill="#FFFFFF", strokeOpacity=0).configure_axis(
        labelColor="#52645A", titleColor="#52645A", tickColor="#DCE5DC"
    )
    st.altair_chart(grafico_resultados, use_container_width=True)

    st.subheader("Matriz de confusión")
    st.caption("Las filas son las categorías reales; las columnas, las predichas. La diagonal contiene los aciertos.")
    matriz_df = pd.DataFrame(matriz, index=categorias, columns=categorias)
    matriz_df.index.name = "Categoría real"
    st.dataframe(matriz_df, alt="Matriz que muestra aciertos y confusiones por categoría")
    st.caption("La evaluación es inicial: el dataset es pequeño y las normas pueden variar según el municipio.")

st.markdown('<div class="eco-footer">Proyecto educativo · Orientación de reciclaje</div>', unsafe_allow_html=True)

st.divider()
st.button("Salir", on_click=cerrar_sesion, help="Cierra esta sesión y borra sus datos.")
st.caption(
    "La predicción es orientativa: el modelo se entrenó con un dataset educativo pequeño. "
    "En España, el sistema de recogida y algunas instrucciones pueden variar según el municipio. "
    "Para residuos especiales o voluminosos, consulta el servicio local o un punto limpio."
)
