import ast

import pandas as pd
import streamlit as st

from analizar_articulos import (
    MODELO,
    cargar_claves,
    crear_prompt,
    solicitar_analisis,
    validar_analisis,
)


ARCHIVO_CSV = "evaluacion_articulos_jaione_sanz.csv"
COLUMNAS_OBLIGATORIAS = [
    "titulo",
    "resumen_tematico",
    "estilo_predominante",
    "claridad_y_estructura",
    "rigor_y_argumentacion",
    "puntos_fuertes",
    "puntos_mejora",
]


st.set_page_config(page_title="Dashboard de Artículos", layout="wide")


def convertir_lista(valor):
    """Convierte una lista guardada como texto en una lista de Python."""
    if not isinstance(valor, str):
        return valor

    try:
        resultado = ast.literal_eval(valor)
        return resultado if isinstance(resultado, list) else [valor]
    except (ValueError, SyntaxError):
        return [valor]


def formatear_fecha(valor) -> str:
    fecha = pd.to_datetime(valor, errors="coerce")

    if pd.isna(fecha):
        return "N/A"

    return fecha.strftime("%d/%m/%Y")


def leer_borrador(archivo) -> str:
    contenido = archivo.getvalue()

    try:
        return contenido.decode("utf-8")
    except UnicodeDecodeError:
        return contenido.decode("latin-1")


@st.cache_data
def cargar_datos(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Convertimos las columnas de texto a listas para mostrarlas como viñetas.
    for columna in ["puntos_fuertes", "puntos_mejora"]:
        if columna in df.columns:
            df[columna] = df[columna].apply(convertir_lista)

    # Convertimos las puntuaciones a valores numéricos.
    for columna in ["claridad_y_estructura", "rigor_y_argumentacion"]:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")

    return df


try:
    df = cargar_datos(ARCHIVO_CSV)
except FileNotFoundError:
    st.error(f"No se encontró '{ARCHIVO_CSV}'. Genera el archivo previamente.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error(f"El archivo '{ARCHIVO_CSV}' está vacío.")
    st.stop()

columnas_faltantes = [
    columna for columna in COLUMNAS_OBLIGATORIAS if columna not in df.columns
]
if columnas_faltantes:
    st.error(
        "Faltan columnas obligatorias en el CSV: "
        + ", ".join(columnas_faltantes)
    )
    st.stop()

df = df.dropna(subset=["claridad_y_estructura", "rigor_y_argumentacion"])
if df.empty:
    st.warning("No hay artículos con puntuaciones válidas para mostrar.")
    st.stop()

# Configuramos los filtros de búsqueda en la barra lateral.
st.sidebar.header("Filtros de búsqueda")

search_query = st.sidebar.text_input("Buscar por título o resumen:")

estilos_disponibles = sorted(
    df["estilo_predominante"].dropna().unique().tolist()
)
estilos_sel = st.sidebar.multiselect(
    "Estilo:", estilos_disponibles, default=estilos_disponibles
)

min_c = float(df["claridad_y_estructura"].min())
max_c = float(df["claridad_y_estructura"].max())
r_claridad = st.sidebar.slider(
    "Claridad y estructura:",
    min_c,
    max_c,
    (min_c, max_c),
    step=0.5,
)

min_r = float(df["rigor_y_argumentacion"].min())
max_r = float(df["rigor_y_argumentacion"].max())
r_rigor = st.sidebar.slider(
    "Agudeza crítica:",
    min_r,
    max_r,
    (min_r, max_r),
    step=0.5,
)

# Aplicamos primero los filtros de estilo y puntuaciones.
df_filtered = df[
    (df["estilo_predominante"].isin(estilos_sel))
    & (df["claridad_y_estructura"].between(*r_claridad))
    & (df["rigor_y_argumentacion"].between(*r_rigor))
]

# Aplicamos después la búsqueda de texto sobre los resultados anteriores.
if search_query:
    mask = (
        df_filtered["titulo"].str.contains(
            search_query, case=False, na=False, regex=False
        )
        | df_filtered["resumen_tematico"].str.contains(
            search_query, case=False, na=False, regex=False
        )
    )
    df_filtered = df_filtered[mask]

st.title("📊 Análisis cualitativo de artículos")

with st.expander("✍️ Analizar un borrador nuevo"):
    archivo_borrador = st.file_uploader(
        "Sube el borrador",
        type=["txt", "md"],
        help="El archivo se analiza, pero no se añade a los rankings ni a los CSV.",
    )
    titulo_borrador = st.text_input("Título del borrador")

    if archivo_borrador is not None:
        texto_borrador = leer_borrador(archivo_borrador)
        st.caption(f"Longitud: {len(texto_borrador):,} caracteres")
    else:
        texto_borrador = ""

    analizar_borrador = st.button(
        "Analizar borrador",
        disabled=not texto_borrador.strip(),
        type="primary",
    )

    if analizar_borrador:
        titulo = titulo_borrador.strip() or archivo_borrador.name
        prompt = crear_prompt(titulo, texto_borrador)
        ultimo_error = None

        with st.spinner(f"Analizando el borrador con {MODELO}..."):
            for clave in cargar_claves():
                try:
                    resultado = solicitar_analisis(clave, MODELO, prompt)
                    resultado = validar_analisis(resultado)
                    st.session_state["analisis_borrador"] = resultado
                    ultimo_error = None
                    break
                except Exception as error:
                    ultimo_error = error

        if ultimo_error:
            st.error(f"No se pudo analizar el borrador: {ultimo_error}")

    resultado_borrador = st.session_state.get("analisis_borrador")

    if resultado_borrador:
        st.markdown("### Resultado del borrador")

        borrador_col1, borrador_col2, borrador_col3 = st.columns(3)
        borrador_col1.metric(
            "Claridad",
            f"{resultado_borrador['claridad_y_estructura']:g} / 10",
        )
        borrador_col2.metric(
            "Agudeza crítica",
            f"{resultado_borrador['rigor_y_argumentacion']:g} / 10",
        )
        borrador_col3.metric(
            "Estilo",
            resultado_borrador["estilo_predominante"],
        )

        st.markdown(
            f"**Resumen:** {resultado_borrador['resumen_tematico']}"
        )

        fortalezas, mejoras = st.columns(2)

        with fortalezas:
            st.markdown("**Puntos fuertes:**")
            for punto in resultado_borrador["puntos_fuertes"]:
                st.markdown(f"- {punto}")

        with mejoras:
            st.markdown("**Puntos de mejora:**")
            for punto in resultado_borrador["puntos_mejora"]:
                st.markdown(f"- {punto}")

st.divider()

col1, col2, col3 = st.columns(3)
col1.metric("Artículos filtrados", len(df_filtered))
col2.metric(
    "Promedio claridad",
    f"{df_filtered['claridad_y_estructura'].mean():.1f} / 10"
    if not df_filtered.empty
    else "N/A",
)
col3.metric(
    "Promedio agudeza crítica",
    f"{df_filtered['rigor_y_argumentacion'].mean():.1f} / 10"
    if not df_filtered.empty
    else "N/A",
)

st.divider()

# Calculamos una puntuación global para crear los rankings.
if not df_filtered.empty:
    df_ranking = df_filtered.copy()
    df_ranking["puntuacion_global"] = df_ranking[
        ["claridad_y_estructura", "rigor_y_argumentacion"]
    ].mean(axis=1)

    columnas_ranking = [
        "titulo",
        "estilo_predominante",
        "claridad_y_estructura",
        "rigor_y_argumentacion",
        "puntuacion_global",
    ]

    if "fecha" in df_ranking.columns:
        df_ranking["fecha"] = df_ranking["fecha"].apply(formatear_fecha)
        columnas_ranking.insert(1, "fecha")

    top_5 = df_ranking.nlargest(5, "puntuacion_global")[columnas_ranking]
    peores_5 = df_ranking.nsmallest(5, "puntuacion_global")[columnas_ranking]

    top_5 = top_5.rename(
        columns={
            "titulo": "Título",
            "fecha": "Fecha",
            "estilo_predominante": "Estilo",
            "claridad_y_estructura": "Claridad",
            "rigor_y_argumentacion": "Agudeza crítica",
            "puntuacion_global": "Puntuación global",
        }
    )
    peores_5 = peores_5.rename(
        columns={
            "titulo": "Título",
            "fecha": "Fecha",
            "estilo_predominante": "Estilo",
            "claridad_y_estructura": "Claridad",
            "rigor_y_argumentacion": "Agudeza crítica",
            "puntuacion_global": "Puntuación global",
        }
    )

    top_5["Puntuación global"] = top_5["Puntuación global"].round(1)
    peores_5["Puntuación global"] = peores_5["Puntuación global"].round(1)

    st.subheader("Ranking de artículos")
    st.caption("Puntuación global = media de claridad y agudeza crítica.")

    st.markdown("#### 🏆 Los 5 mejores")
    st.dataframe(top_5, hide_index=True, width="stretch")

    st.markdown("#### 📉 Los 5 peores")
    st.dataframe(peores_5, hide_index=True, width="stretch")

    st.divider()

if df_filtered.empty:
    st.info("No hay artículos que coincidan con los filtros seleccionados.")

st.subheader("Listado y desglose de evaluaciones")

for _, row in df_filtered.iterrows():
    fecha = formatear_fecha(row.get("fecha"))
    header_text = (
        f"{fecha} | {row['titulo']} | "
        f"Claridad: {row['claridad_y_estructura']:g}/10 | "
        f"Agudeza crítica: {row['rigor_y_argumentacion']:g}/10"
    )

    with st.expander(header_text):
        st.caption(f"Fecha: {fecha} | Estilo: {row['estilo_predominante']}")
        st.markdown(f"**Resumen:** {row['resumen_tematico']}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Puntos fuertes:**")
            for item in row["puntos_fuertes"]:
                st.markdown(f"- {item}")

        with c2:
            st.markdown("**Puntos de mejora:**")
            for item in row["puntos_mejora"]:
                st.markdown(f"- {item}")

        if "url" in df.columns and pd.notna(row.get("url")):
            st.markdown(f"[🔗 Enlace al artículo original]({row['url']})")
