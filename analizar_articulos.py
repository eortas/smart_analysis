import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

from limpieza_texto import limpiar_texto


ARCHIVO_ENTRADA = "articulos_jaione_sanz.csv"
ARCHIVO_SALIDA = "evaluacion_articulos_jaione_sanz.csv"
URL_API = "https://api.mistral.ai/v1/chat/completions"
URL_API_GROQ = "https://api.groq.com/openai/v1/chat/completions"

# Evaluamos y reescribimos con qwen/qwen3.8-27b en Groq, manteniendo
# alineados el criterio editorial de análisis y la propuesta de reescritura.
VERSION_CRITERIO = "opinion_qwen_v2"

MODELO = "qwen/qwen3.8-27b"

# Reescritura de borradores: mismo modelo que la evaluación (ver arriba).
MODELO_REESCRITURA = MODELO

CAMPOS_ANALISIS = [
    "resumen_tematico",
    "estilo_predominante",
    "critica_editorial",
    "claridad_y_estructura",
    "rigor_y_argumentacion",
    "puntos_fuertes",
    "puntos_mejora",
]


def cargar_claves() -> list[str]:
    load_dotenv()

    claves = [os.getenv("MISTRAL1"), os.getenv("MISTRAL2")]
    claves = [clave for clave in claves if clave]

    if not claves:
        raise ValueError("No se encontraron MISTRAL1 o MISTRAL2 en el archivo .env.")

    return claves


def cargar_claves_groq() -> list[str]:
    """Carga todas las claves de Groq disponibles para alternarlas por artículo."""
    load_dotenv()

    candidatas = [
        os.getenv("GROQ1"),
        os.getenv("GROQ2"),
        os.getenv("GROQ3"),
        os.getenv("GROQ_API_KEY"),
    ]
    claves = []
    for clave in candidatas:
        if clave and clave not in claves:
            claves.append(clave)

    if not claves:
        raise ValueError(
            "No se encontraron claves de Groq (GROQ1, GROQ2, GROQ3 o GROQ_API_KEY) en el .env."
        )

    return claves


def es_modelo_groq(modelo: str) -> bool:
    """True si el ID corresponde a un modelo alojado en Groq."""
    return modelo.startswith(
        ("openai/", "qwen/", "meta-llama/", "allam-", "moonshotai/")
    )


def claves_para_modelo(modelo: str) -> list[str]:
    """Devuelve las claves adecuadas para el proveedor del modelo."""
    if es_modelo_groq(modelo):
        return cargar_claves_groq()
    return cargar_claves()


def crear_prompt(titulo: str, texto: str) -> str:
    return f"""
Analiza el siguiente artículo de opinión en español.

Valora el artículo como un editor experto en periodismo y en comunicación.
Sé estrictamente profesional y valora los artículos de opinión con ese rigor.

Devuelve solamente un objeto JSON con esta estructura exacta:
{{
  "resumen_tematico": "Resumen neutral de 2 o 3 frases",
  "estilo_predominante": "Una categoría breve",
  "critica_editorial": "2 o 3 frases con observaciones críticas constructivas sobre el texto",
  "claridad_y_estructura": 7.5,
  "rigor_y_argumentacion": 7.5,
  "puntos_fuertes": ["Punto concreto", "Punto concreto"],
  "puntos_mejora": ["Punto concreto", "Punto concreto"]
}}

Las puntuaciones deben estar entre 1 y 10 y utilizar incrementos de 0.5.
Usa activamente los medios puntos cuando la valoración quede entre dos niveles.
Evalúa únicamente el contenido recibido y no inventes datos externos.
Incluye entre 2 y 4 elementos en cada lista.

En "critica_editorial", resume en 2 o 3 frases las principales observaciones
críticas y oportunidades de mejora sobre el texto. Toda columna periodística,
incluso una excelente, admite matices o sugerencias de pulido; estas observaciones
no impiden calificar con notas altas (9.0 o 9.5) a piezas con gran brillo literario y fuerza crítica.

Para "claridad_y_estructura", valora la claridad de la tesis, la organización
de las ideas, el ritmo narrativo y la fluidez de lectura.

Para "rigor_y_argumentacion", no exijas el rigor de una investigación académica.
Valora la agudeza y eficacia de la crítica: coherencia de la opinión, fuerza de
los argumentos, originalidad, ironía, humor socarrón, sátira, intención
provocadora y buen uso de recursos retóricos. No penalices la falta de citas o
datos si el texto funciona con maestría como columna de opinión.

En "estilo_predominante" puedes utilizar categorías como irónico, socarrón,
satírico, crítico, reflexivo, provocador, humorístico o combativo.

Los puntos fuertes y de mejora deben juzgar el texto como artículo de opinión.

Calibración de notas (escala de 1 a 10, siempre en saltos de 0.5):
Utiliza toda la escala con naturalidad, reconociendo el mérito de las mejores piezas:
- 9.0 a 9.5 (o 10): Columna sobresaliente o brillante. Tesis original o audaz, gran pulso narrativo, voz propia muy reconocible, ironía eficaz y cierre redondo. No temas otorgar 9.0 o 9.5 a las columnas que destaquen por su agudeza y maestría en la escritura.
- 8.0 a 8.5: Columna notable, sólida, bien armada y con buena pegada persuasiva.
- 7.0 a 7.5: Columna correcta y funcional, con ritmo adecuado pero sin especial brillo.
- 5.5 a 6.5: Columna irregular o plana, con saltos de tono, tópicos o desarrollo disperso.
- 1.0 a 5.0: Columna deficiente, superficial o fallida.

Tener observaciones críticas o puntos de mejora es propio de cualquier ejercicio editorial y es perfectamente compatible con una nota de 9.0 o 9.5 si el artículo es excelente en su género.

Título: {titulo}

Artículo:
{texto}
""".strip()


def solicitar_analisis(
    clave: str,
    modelo: str,
    prompt: str,
    max_intentos: int = 4,
    max_tokens: int | None = None,
) -> dict:
    headers = {
        "Authorization": f"Bearer {clave}",
        "Content-Type": "application/json",
    }
    url = URL_API_GROQ if es_modelo_groq(modelo) else URL_API

    datos = {
        "model": modelo,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Eres especialista en análisis de textos periodísticos. "
                    "Respondes siempre en español y en JSON válido."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "response_format": {"type": "json_object"},
    }

    # Las reescrituras pueden ser largas: permitimos subir el toque de salida.
    if max_tokens:
        datos["max_tokens"] = max_tokens

    for intento in range(1, max_intentos + 1):
        respuesta = requests.post(
            url,
            headers=headers,
            json=datos,
            timeout=120,
        )

        if respuesta.status_code == 200:
            contenido = respuesta.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(contenido)
            except json.JSONDecodeError:
                # El modelo pequeño a veces emite JSON mal formado: reintentamos.
                if intento < max_intentos:
                    print("Respuesta no válida en JSON. Reintentando.")
                    time.sleep(2)
                    continue
                raise

        if respuesta.status_code == 429 and intento < max_intentos:
            espera = int(respuesta.headers.get("Retry-After", intento * 10))
            print(f"Límite temporal alcanzado. Esperamos {espera} segundos.")
            time.sleep(espera)
            continue

        # Groq valida el JSON en servidor y a veces rechaza una generación
        # por azar de la muestra: merece un reintento como el 429.
        if (
            "json_validate_failed" in respuesta.text
            and intento < max_intentos
        ):
            print("El proveedor rechazó la generación JSON. Reintentando.")
            time.sleep(2)
            continue

        mensaje = respuesta.text[:500]
        raise requests.HTTPError(
            f"El proveedor devolvió {respuesta.status_code}: {mensaje}",
            response=respuesta,
        )

    raise RuntimeError("No se pudo obtener el análisis después de varios intentos.")


def _a_texto(valor) -> str:
    """Coacciona campos de texto: el modelo a veces devuelve dicts o listas."""
    if isinstance(valor, str):
        texto = valor.strip()
        if texto.startswith("[") and texto.endswith("]"):
            try:
                interno = json.loads(texto)
            except (json.JSONDecodeError, ValueError):
                interno = None
            if isinstance(interno, list):
                return "; ".join(str(elemento) for elemento in interno)
        return valor
    if isinstance(valor, list):
        return "; ".join(str(elemento) for elemento in valor)
    if isinstance(valor, dict):
        return " ".join(str(valor_campo) for valor_campo in valor.values())
    return str(valor)


def validar_analisis(analisis: dict) -> dict:
    faltantes = [campo for campo in CAMPOS_ANALISIS if campo not in analisis]
    if faltantes:
        raise ValueError(f"Faltan campos en el análisis: {', '.join(faltantes)}")

    for campo in ["resumen_tematico", "estilo_predominante", "critica_editorial"]:
        analisis[campo] = _a_texto(analisis[campo])

    # Redondeamos las notas al medio punto más cercano.
    claridad = round(float(analisis["claridad_y_estructura"]) * 2) / 2
    agudeza = round(float(analisis["rigor_y_argumentacion"]) * 2) / 2

    analisis["claridad_y_estructura"] = max(1.0, min(10.0, claridad))
    analisis["rigor_y_argumentacion"] = max(1.0, min(10.0, agudeza))

    for campo in ["puntos_fuertes", "puntos_mejora"]:
        if not isinstance(analisis[campo], list):
            analisis[campo] = [str(analisis[campo])]

    return analisis


def crear_prompt_reescritura(titulo: str, texto: str, analisis: dict) -> str:
    fortalezas = "\n".join(
        f"- {punto}" for punto in analisis.get("puntos_fuertes", [])
    )
    mejoras = "\n".join(
        f"- {punto}" for punto in analisis.get("puntos_mejora", [])
    )
    critica = analisis.get("critica_editorial", "")

    return f"""
Reescribe el siguiente borrador de artículo de opinión aplicando las mejoras
señaladas en su evaluación editorial, con criterio de editor experto en
periodismo y en comunicación. Sé estrictamente profesional.

Reglas:
- Corrige los fallos indicados en la crítica editorial y aplica cada uno de
  los puntos de mejora.
- Conserva la voz, el tono y la extensión aproximada del texto original.
  No reescribas desde cero: edita con criterio.
- Extensión obligatoria: la reescritura debe tener entre
  {int(len(texto) * 0.85):,} y {int(len(texto) * 1.15):,} caracteres
  (el original tiene {len(texto):,}). No amplíes el texto: desarrolla las
  ideas dentro de esa extensión.
- Mantén los puntos fuertes; no los diluyas ni los elimines.
- No inventes datos ni citas nuevas. Si falta información, trabaja con lo que
  el propio texto aporta.
- Devuelve un texto pulido, listo para publicar.

El JSON debe ser válido: escapa las comillas dobles y los saltos de línea
dentro de los valores de texto. El campo "reescritura" contiene el texto
completo, sin recortar.

Devuelve solamente un objeto JSON con esta estructura exacta:
{{
  "reescritura": "El texto completo reescrito",
  "cambios_realizados": ["Cambio concreto aplicado", "Cambio concreto aplicado"]
}}

"cambios_realizados" debe listar de 3 a 6 cambios concretos (qué se corrigió
y por qué), en frases breves.

Título: {titulo}

Crítica editorial de la evaluación:
{critica}

Puntos fuertes a conservar:
{fortalezas}

Puntos de mejora a aplicar:
{mejoras}

Borrador original:
{texto}
""".strip()


def validar_reescritura(
    reescritura: dict,
    texto_original: str = "",
    estricto: bool = True,
) -> dict:
    faltantes = [
        campo for campo in ("reescritura", "cambios_realizados")
        if campo not in reescritura
    ]
    if faltantes:
        raise ValueError(
            f"Faltan campos en la reescritura: {', '.join(faltantes)}"
        )

    texto = str(reescritura["reescritura"]).strip()
    if not texto:
        raise ValueError("La reescritura llegó vacía.")

    # Control de truncado o expansión descontrolada: la reescritura debe
    # acercarse a la extensión del original (recorrido 0.5x–1.6x).
    # Con estricto=False solo se valida la estructura (para el fallback).
    if texto_original and estricto:
        if len(texto) < len(texto_original) * 0.5:
            raise ValueError(
                "La reescritura es demasiado corta respecto al original; "
                "probablemente llegó incompleta."
            )
        if len(texto) > len(texto_original) * 1.6:
            raise ValueError(
                "La reescritura se expandió demasiado respecto al original; "
                "se pidió conservar la extensión."
            )

    cambios = reescritura["cambios_realizados"]
    if not isinstance(cambios, list):
        cambios = [str(cambios)]
    reescritura["reescritura"] = texto
    reescritura["cambios_realizados"] = [str(cambio) for cambio in cambios]

    return reescritura


def reescribir_borrador(
    claves: list[str],
    modelo: str,
    titulo: str,
    texto: str,
    analisis: dict,
    max_rondas: int = 2,
) -> dict:
    """Reescribe el borrador con reintentos sobre todas las claves.

    Si tras varias rondas el modelo se empeña en desbordar la extensión
    pedida, devuelve el mejor intento con un campo "aviso_extension" para
    que el dashboard lo muestre en vez de fallar del todo.
    """
    prompt = crear_prompt_reescritura(titulo, texto, analisis)
    ultimo_error = None
    mejor_excedido = None
    mejor_ratio = float("inf")

    for _ in range(max_rondas):
        for clave in claves:
            try:
                crudo = solicitar_analisis(
                    clave, modelo, prompt, max_tokens=6000
                )
            except Exception as error:
                ultimo_error = error
                continue

            try:
                return validar_reescritura(crudo, texto, estricto=True)
            except ValueError as error:
                ultimo_error = error
                try:
                    posible = validar_reescritura(
                        crudo, texto, estricto=False
                    )
                    ratio = len(posible["reescritura"]) / max(len(texto), 1)
                    if ratio < mejor_ratio:
                        mejor_ratio = ratio
                        mejor_excedido = posible
                except ValueError:
                    pass

    if mejor_excedido is not None:
        mejor_excedido["aviso_extension"] = (
            f"La reescritura quedó un {int((mejor_ratio - 1) * 100):+d}% "
            "más larga de lo pedido: revisa si te conviene recortarla."
        )
        return mejor_excedido

    raise ultimo_error or RuntimeError(
        "No se pudo reescribir el borrador."
    )



def seleccionar_modelo(claves: list[str]) -> tuple[str, str]:
    prompt_prueba = (
        'Devuelve solamente este JSON: {"ok": true}. '
        "No añadas ningún otro texto."
    )

    for clave in claves:
        try:
            solicitar_analisis(
                clave,
                MODELO,
                prompt_prueba,
                max_intentos=1,
            )
            print(f"Modelo seleccionado: {MODELO}")
            return MODELO, clave
        except (requests.RequestException, json.JSONDecodeError, KeyError):
            continue

    raise RuntimeError(
        f"Ninguna clave permite usar el modelo '{MODELO}'."
    )


def cargar_resultados_anteriores() -> list[dict]:
    if not os.path.exists(ARCHIVO_SALIDA):
        return []

    df_anterior = pd.read_csv(ARCHIVO_SALIDA)

    if "url" in df_anterior.columns:
        df_anterior = df_anterior.drop_duplicates(subset="url", keep="last")

    return df_anterior.to_dict("records")


def guardar_resultados(resultados: list[dict]):
    df_resultados = pd.DataFrame(resultados)

    if not df_resultados.empty and "url" in df_resultados.columns:
        df_resultados = df_resultados.drop_duplicates(subset="url", keep="last")

    if not df_resultados.empty and "fecha" in df_resultados.columns:
        # Ordenamos los resultados por fecha de forma descendente
        fechas_ord = pd.to_datetime(
            df_resultados["fecha"], errors="coerce", utc=True
        )
        df_resultados = (
            df_resultados.assign(_fecha_ord=fechas_ord)
            .sort_values("_fecha_ord", ascending=False, na_position="last")
            .drop(columns="_fecha_ord")
        )

    df_resultados.to_csv(
        ARCHIVO_SALIDA,
        index=False,
        encoding="utf-8-sig",
    )


def ejecutar_analisis():
    if not os.path.exists(ARCHIVO_ENTRADA):
        raise FileNotFoundError(
            f"No existe '{ARCHIVO_ENTRADA}'. Ejecuta primero scraper.py."
        )

    # Con un modelo de Groq carga las claves GROQ1/GROQ2 (y con un modelo
    # de Mistral, MISTRAL1/MISTRAL2).
    claves = claves_para_modelo(MODELO)
    modelo, clave_principal = seleccionar_modelo(claves)
    claves_ordenadas = [clave_principal] + [
        clave for clave in claves if clave != clave_principal
    ]

    df = pd.read_csv(ARCHIVO_ENTRADA)
    resultados = cargar_resultados_anteriores()
    urls_analizadas = {
        fila.get("url")
        for fila in resultados
        if fila.get("version_criterio") == VERSION_CRITERIO
    }

    pendientes = df[~df["url"].isin(urls_analizadas)]

    if not pendientes.empty and "fecha" in pendientes.columns:
        # Priorizamos el análisis de los artículos más recientes
        fechas_ord = pd.to_datetime(
            pendientes["fecha"], errors="coerce", utc=True
        )
        pendientes = (
            pendientes.assign(_fecha_ord=fechas_ord)
            .sort_values("_fecha_ord", ascending=False, na_position="last")
            .drop(columns="_fecha_ord")
        )

    print(f"Artículos pendientes: {len(pendientes)}")

    for numero, (_, articulo) in enumerate(pendientes.iterrows(), start=1):
        print(f"Analizando {numero}/{len(pendientes)}: {articulo['titulo']}")
        # Los textos guardados antes de corregir el scraper incluían los
        # fragmentos destacados (pull quotes) duplicados; sin esta limpieza
        # el modelo interpreta la repetición como un fallo del artículo.
        texto = limpiar_texto(articulo["texto"])
        prompt = crear_prompt(articulo["titulo"], texto)
        ultimo_error = None

        # Alternamos la clave preferida y conservamos la otra como respaldo.
        indice_clave = (numero - 1) % len(claves_ordenadas)
        claves_articulo = (
            claves_ordenadas[indice_clave:]
            + claves_ordenadas[:indice_clave]
        )

        for clave in claves_articulo:
            try:
                analisis = solicitar_analisis(clave, modelo, prompt)
                analisis = validar_analisis(analisis)
                resultado = articulo.to_dict() | analisis
                resultado["texto"] = texto
                resultado["longitud_caracteres"] = len(texto)
                resultado["version_criterio"] = VERSION_CRITERIO
                resultados.append(resultado)
                guardar_resultados(resultados)
                ultimo_error = None
                break
            except (
                requests.RequestException,
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as error:
                ultimo_error = error

        if ultimo_error:
            print(f"No se pudo analizar el artículo: {ultimo_error}")

        time.sleep(4)

    print(f"CSV generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_analisis()
