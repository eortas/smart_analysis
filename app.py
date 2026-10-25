import ast
import os

import pandas as pd
import streamlit as st

from analizar_articulos import (
    EVALUACIONES_COMPARATIVAS,
    MODELO,
    MODELO_REESCRITURA,
    TONOS_CRITICA,
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
    columnas_lista = [
        "puntos_fuertes",
        "puntos_mejora",
        "puntos_fuertes_clasico",
        "puntos_mejora_clasico",
    ]
    for columna in columnas_lista:
        if columna in df.columns:
            df[columna] = df[columna].apply(convertir_lista)

    # Convertimos las puntuaciones a valores numéricos.
    columnas_numericas = [
        "claridad_y_estructura",
        "rigor_y_argumentacion",
        "claridad_y_estructura_clasico",
        "rigor_y_argumentacion_clasico",
    ]
    for columna in columnas_numericas:
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

with st.expander("Analizar un borrador nuevo (con selector de tono)", expanded=True):
    col_subida, col_tono = st.columns([1, 1])

    with col_subida:
        archivo_borrador = st.file_uploader(
            "Sube el borrador",
            type=["txt", "md"],
            help="El archivo se analiza, pero no se añade a los rankings ni a los CSV.",
        )
        titulo_borrador = st.text_input("Título del borrador")

    with col_tono:
        # Añadimos el selector interactivo para alternar entre el tono desenfadado y el editorial
        opciones_tono = list(TONOS_CRITICA.keys())
        tono_seleccionado = st.radio(
            "Tono de la crítica editorial:",
            options=opciones_tono,
            format_func=lambda t: TONOS_CRITICA[t]["nombre"],
            index=0,
            help="Elige si prefieres una crítica cómplice en la voz de Jaione Sanz o una revisión editorial clásica.",
        )
        st.caption(f"ℹ️ {TONOS_CRITICA[tono_seleccionado]['descripcion']}")

    if archivo_borrador is not None:
        texto_borrador = leer_borrador(archivo_borrador)
        st.caption(f"Longitud: {len(texto_borrador):,} caracteres")
    else:
        # Permitimos pegar el texto directamente si no se dispone de archivo
        texto_pegado = st.text_area(
            "O pega el contenido del borrador aquí directamente:",
            height=140,
            placeholder="Pega aquí el texto del artículo para analizarlo...",
        )
        texto_borrador = texto_pegado.strip()
        if texto_borrador:
            st.caption(f"Longitud: {len(texto_borrador):,} caracteres")

    analizar_borrador = st.button(
        "Analizar borrador",
        disabled=not texto_borrador.strip(),
        type="primary",
    )

    if analizar_borrador:
        titulo = (
            titulo_borrador.strip()
            or (archivo_borrador.name if archivo_borrador else "Borrador sin título")
        )
        prompt = crear_prompt(titulo, texto_borrador, tono=tono_seleccionado)
        prompt_sistema = TONOS_CRITICA[tono_seleccionado]["prompt_sistema"]
        ultimo_error = None

        # Un análisis nuevo invalida la reescritura anterior, si existía.
        st.session_state.pop("reescritura_borrador", None)

        with st.spinner("Analizando el borrador con IA personalizada..."):
            for clave in claves_para_modelo(MODELO):
                try:
                    resultado = solicitar_analisis(
                        clave,
                        MODELO,
                        prompt,
                        prompt_sistema=prompt_sistema,
                    )
                    resultado = validar_analisis(resultado)
                    resultado["tono_utilizado"] = TONOS_CRITICA[tono_seleccionado]["nombre"]
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
        if resultado_borrador.get("tono_utilizado"):
            st.caption(f"Enfoque aplicado: {resultado_borrador['tono_utilizado']}")

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

for idx, (_, row) in enumerate(df_filtered.iterrows()):
    fecha = formatear_fecha(row.get("fecha"))
    titulo_art = row["titulo"]

    # Comprobamos si el artículo dispone de comparativa (por diccionario o columnas del CSV)
    tiene_clasico_csv = (
        "critica_editorial_clasico" in df.columns
        and pd.notna(row.get("critica_editorial_clasico"))
        and str(row.get("critica_editorial_clasico")).strip() != ""
    )
    tiene_comparativa = (titulo_art in EVALUACIONES_COMPARATIVAS) or tiene_clasico_csv

    # Destacamos visualmente los artículos con opción de cambiar de tono
    etiqueta_comparativa = " 🎭 (Comparador de tonos disponible)" if tiene_comparativa else ""
    header_text = (
        f"{fecha} | {titulo_art} | "
        f"Claridad: {row['claridad_y_estructura']:g}/10 | "
        f"Agudeza crítica: {row['rigor_y_argumentacion']:g}/10{etiqueta_comparativa}"
    )

    with st.expander(header_text, expanded=tiene_comparativa):
        if tiene_comparativa:
            if titulo_art in EVALUACIONES_COMPARATIVAS:
                datos_comp = EVALUACIONES_COMPARATIVAS[titulo_art]
            else:
                # Construimos la comparativa a partir de las columnas almacenadas en el CSV
                pf_des = row.get("puntos_fuertes", [])
                pm_des = row.get("puntos_mejora", [])
                pf_cla = row.get("puntos_fuertes_clasico", [])
                pm_cla = row.get("puntos_mejora_clasico", [])

                datos_comp = {
                    "desenfadado": {
                        "nombre": "🎭 Nuevo prompt (Voz desenfadada de Jaione Sanz)",
                        "resumen_tematico": str(row.get("resumen_tematico", "")),
                        "estilo_predominante": str(row.get("estilo_predominante", "")),
                        "critica_editorial": str(row.get("critica_editorial", "")),
                        "claridad_y_estructura": row.get("claridad_y_estructura", 0.0),
                        "rigor_y_argumentacion": row.get("rigor_y_argumentacion", 0.0),
                        "puntos_fuertes": pf_des if isinstance(pf_des, list) else [pf_des],
                        "puntos_mejora": pm_des if isinstance(pm_des, list) else [pm_des],
                    },
                    "clasico": {
                        "nombre": "📰 Prompt anterior (Editor profesional clásico)",
                        "resumen_tematico": str(row.get("resumen_tematico_clasico", "")),
                        "estilo_predominante": str(row.get("estilo_predominante_clasico", "")),
                        "critica_editorial": str(row.get("critica_editorial_clasico", "")),
                        "claridad_y_estructura": row.get("claridad_y_estructura_clasico", 0.0),
                        "rigor_y_argumentacion": row.get("rigor_y_argumentacion_clasico", 0.0),
                        "puntos_fuertes": pf_cla if isinstance(pf_cla, list) else [pf_cla],
                        "puntos_mejora": pm_cla if isinstance(pm_cla, list) else [pm_cla],
                    },
                }

            st.markdown("#### 🔄 Comparador de enfoques de crítica editorial")
            modo_tono = st.radio(
                "Selecciona el prompt para este artículo:",
                options=["desenfadado", "clasico", "lado_a_lado"],
                format_func=lambda k: {
                    "desenfadado": "🎭 Nuevo prompt (Voz desenfadada de Jaione Sanz)",
                    "clasico": "📰 Prompt anterior (Editor profesional clásico)",
                    "lado_a_lado": "⚖️ Comparativa lado a lado",
                }[k],
                horizontal=True,
                key=f"selector_tono_art_{idx}",
                help="Alterna entre la crítica con la voz y estilo de Jaione Sanz y la evaluación editorial clásica.",
            )

            if modo_tono in ["desenfadado", "clasico"]:
                item_eval = datos_comp[modo_tono]
                c_m1, c_m2, c_m3 = st.columns(3)
                c_m1.metric("Claridad", f"{item_eval['claridad_y_estructura']} / 10")
                c_m2.metric("Agudeza crítica", f"{item_eval['rigor_y_argumentacion']} / 10")
                c_m3.metric("Estilo", item_eval["estilo_predominante"])

                st.caption(f"Fecha: {fecha} | Enfoque: {item_eval['nombre']}")
                st.markdown(f"**Resumen:** {item_eval['resumen_tematico']}")
                st.markdown(f"**Crítica editorial:** {item_eval['critica_editorial']}")

                col_pf, col_pm = st.columns(2)
                with col_pf:
                    st.markdown("**Puntos fuertes:**")
                    for pf in item_eval["puntos_fuertes"]:
                        st.markdown(f"- {pf}")
                with col_pm:
                    st.markdown("**Puntos de mejora:**")
                    for pm in item_eval["puntos_mejora"]:
                        st.markdown(f"- {pm}")
            else:
                # Mostramos la comparativa directa lado a lado
                st.caption("Comparativa directa entre ambos enfoques para el mismo artículo:")
                col_izq, col_der = st.columns(2)
                with col_izq:
                    st.markdown("##### 🎭 Nuevo prompt (Voz desenfadada)")
                    e_des = datos_comp["desenfadado"]
                    st.markdown(f"**Estilo:** {e_des['estilo_predominante']}")
                    st.markdown(
                        f"**Claridad:** {e_des['claridad_y_estructura']}/10 | "
                        f"**Agudeza:** {e_des['rigor_y_argumentacion']}/10"
                    )
                    st.markdown(f"**Crítica editorial:**\n\n> {e_des['critica_editorial']}")
                    st.markdown("**Puntos fuertes:**")
                    for pf in e_des["puntos_fuertes"]:
                        st.markdown(f"- {pf}")
                    st.markdown("**Puntos de mejora:**")
                    for pm in e_des["puntos_mejora"]:
                        st.markdown(f"- {pm}")

                with col_der:
                    st.markdown("##### 📰 Prompt anterior (Editor clásico)")
                    e_cla = datos_comp["clasico"]
                    st.markdown(f"**Estilo:** {e_cla['estilo_predominante']}")
                    st.markdown(
                        f"**Claridad:** {e_cla['claridad_y_estructura']}/10 | "
                        f"**Agudeza:** {e_cla['rigor_y_argumentacion']}/10"
                    )
                    st.markdown(f"**Crítica editorial:**\n\n> {e_cla['critica_editorial']}")
                    st.markdown("**Puntos fuertes:**")
                    for pf in e_cla["puntos_fuertes"]:
                        st.markdown(f"- {pf}")
                    st.markdown("**Puntos de mejora:**")
                    for pm in e_cla["puntos_mejora"]:
                        st.markdown(f"- {pm}")
        else:
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
