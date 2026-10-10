import ast
import os

import pandas as pd
import streamlit as st

from analizar_articulos import (
    MODELO,
    MODELO_REESCRITURA,
    claves_para_modelo,
    crear_prompt,
    reescribir_borrador,
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
def cargar_datos(filepath: str, _mtime: float = 0.0) -> pd.DataFrame:
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
    # Incluimos la fecha de modificación para invalidar el caché cuando el CSV
    # se regenera (si no, Streamlit seguiría mostrando los datos antiguos).
    df = cargar_datos(ARCHIVO_CSV, os.path.getmtime(ARCHIVO_CSV))
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

st.title("📊 Análisis de artículos con IA")

# Integramos los filtros en la ventana principal.
st.subheader("Filtros de búsqueda")
filtro_busqueda, filtro_estilo = st.columns(2)

with filtro_busqueda:
    search_query = st.text_input("Buscar por título o resumen:")

estilos_disponibles = sorted(
    df["estilo_predominante"].dropna().unique().tolist()
)
with filtro_estilo:
    estilos_sel = st.multiselect(
        "Estilo:", estilos_disponibles, default=estilos_disponibles
    )

# Aplicamos el filtro de estilo.
df_filtered = df[df["estilo_predominante"].isin(estilos_sel)]

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

# Ordenamos de la publicación más reciente a la más antigua: el orden del
# CSV es el del scraping, no el cronológico (la última columna aparece
# la primera y viceversa).
if "fecha" in df_filtered.columns:
    fechas_ord = pd.to_datetime(
        df_filtered["fecha"], errors="coerce", utc=True
    )
    df_filtered = (
        df_filtered.assign(_fecha_ord=fechas_ord)
        .sort_values("_fecha_ord", ascending=False, na_position="last")
        .drop(columns="_fecha_ord")
    )

with st.expander("Analizar un borrador nuevo"):
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

        # Un análisis nuevo invalida la reescritura anterior, si existía.
        st.session_state.pop("reescritura_borrador", None)

        with st.spinner("Analizando el borrador con IA personalizada..."):
            for clave in claves_para_modelo(MODELO):
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

        if resultado_borrador.get("critica_editorial"):
            st.markdown(
                f"**Crítica editorial:** {resultado_borrador['critica_editorial']}"
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

        boton_reescribir = st.button(
            "Reescribir borrador con las mejoras sugeridas",
            disabled=not texto_borrador.strip(),
        )

        if boton_reescribir:
            titulo = titulo_borrador.strip() or archivo_borrador.name
            ultimo_error = None

            with st.spinner(
                "Reescribiendo el borrador con IA personalizada..."
            ):
                try:
                    reescritura = reescribir_borrador(
                        claves_para_modelo(MODELO_REESCRITURA),
                        MODELO_REESCRITURA,
                        titulo,
                        texto_borrador,
                        resultado_borrador,
                    )
                    st.session_state["reescritura_borrador"] = reescritura
                    ultimo_error = None
                except Exception as error:
                    ultimo_error = error

            if ultimo_error:
                st.error(f"No se pudo reescribir el borrador: {ultimo_error}")

        reescritura_borrador = st.session_state.get("reescritura_borrador")

        if reescritura_borrador:
            st.markdown("### Borrador reescrito")

            if reescritura_borrador.get("aviso_extension"):
                st.warning(reescritura_borrador["aviso_extension"])

            st.markdown("**Cambios aplicados:**")
            for cambio in reescritura_borrador["cambios_realizados"]:
                st.markdown(f"- {cambio}")

            col_original, col_reescrito = st.columns(2)

            with col_original:
                st.markdown("**Original**")
                st.text_area(
                    "Texto original",
                    texto_borrador,
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                )

            with col_reescrito:
                st.markdown("**Reescrito**")
                st.text_area(
                    "Texto reescrito",
                    reescritura_borrador["reescritura"],
                    height=420,
                    disabled=True,
                    label_visibility="collapsed",
                )

            st.download_button(
                "Descargar borrador reescrito",
                data=reescritura_borrador["reescritura"],
                file_name=(
                    f"{titulo_borrador.strip() or 'borrador'}_reescrito.md"
                ),
                mime="text/markdown",
            )

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

# Mostramos la evolución temporal de las puntuaciones.
if not df_filtered.empty and "fecha" in df_filtered.columns:
    df_evolucion = df_filtered.copy()
    df_evolucion["Fecha"] = pd.to_datetime(
        df_evolucion["fecha"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)
    df_evolucion = df_evolucion.dropna(subset=["Fecha"])

    if not df_evolucion.empty:
        df_evolucion["Puntuación global"] = df_evolucion[
            ["claridad_y_estructura", "rigor_y_argumentacion"]
        ].mean(axis=1)
        df_evolucion = df_evolucion.rename(
            columns={
                "claridad_y_estructura": "Claridad",
                "rigor_y_argumentacion": "Agudeza crítica",
            }
        ).sort_values("Fecha")

        st.subheader("Evolución de las puntuaciones")
        st.line_chart(
            df_evolucion,
            x="Fecha",
            y=["Puntuación global", "Claridad", "Agudeza crítica"],
            color=["#2563EB", "#16A34A", "#EA580C"],
            height=400,
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

        if "critica_editorial" in df.columns and pd.notna(
            row.get("critica_editorial")
        ):
            st.markdown(f"**Crítica editorial:** {row['critica_editorial']}")

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
