import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


ARCHIVO_ENTRADA = "articulos_jaione_sanz.csv"
ARCHIVO_SALIDA = "evaluacion_articulos_jaione_sanz.csv"
URL_API = "https://api.mistral.ai/v1/chat/completions"
VERSION_CRITERIO = "opinion_v3"

MODELO = "mistral-large-latest"

CAMPOS_ANALISIS = [
    "resumen_tematico",
    "estilo_predominante",
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


def crear_prompt(titulo: str, texto: str) -> str:
    return f"""
Analiza el siguiente artículo de opinión en español.

Devuelve solamente un objeto JSON con esta estructura exacta:
{{
  "resumen_tematico": "Resumen neutral de 2 o 3 frases",
  "estilo_predominante": "Una categoría breve",
  "claridad_y_estructura": 6.5,
  "rigor_y_argumentacion": 7.5,
  "puntos_fuertes": ["Punto concreto", "Punto concreto"],
  "puntos_mejora": ["Punto concreto", "Punto concreto"]
}}

Las puntuaciones deben estar entre 1 y 10 y utilizar incrementos de 0.5.
Usa activamente los medios puntos cuando la valoración quede entre dos niveles.
Evalúa únicamente el contenido recibido y no inventes datos externos.
Incluye entre 2 y 4 elementos en cada lista.

Para "claridad_y_estructura", valora la claridad de la tesis, la organización
de las ideas, el ritmo y la facilidad de lectura.

Para "rigor_y_argumentacion", no exijas el rigor de una investigación académica.
Valora la agudeza y eficacia de la crítica: coherencia de la opinión, fuerza de
los argumentos, originalidad, ironía, humor socarrón, sátira, intención
provocadora y buen uso de recursos retóricos. No penalices la falta de citas o
datos si el texto funciona correctamente como columna de opinión.

En "estilo_predominante" puedes utilizar categorías como irónico, socarrón,
satírico, crítico, reflexivo, provocador, humorístico o combativo.

Los puntos fuertes y de mejora deben juzgar el texto como artículo de opinión.

Utiliza toda la escala y evita concentrar las notas entre 7 y 9:
- 9 a 10: pieza excepcional, muy original y prácticamente sin debilidades.
- 7 a 8.5: pieza buena, pero con limitaciones claras.
- 5 a 6.5: pieza correcta o irregular, con virtudes y defectos importantes.
- 3 a 4.5: pieza débil, superficial, repetitiva o poco convincente.
- 1 a 2.5: pieza muy deficiente o incoherente.

Una publicación profesional no merece automáticamente una nota alta. Penaliza
los tópicos, la reiteración, la falta de una tesis clara, los saltos lógicos,
la ironía forzada, la crítica superficial y los finales poco efectivos.
Los valores 9 y 10 deben ser poco frecuentes.

Título: {titulo}

Artículo:
{texto}
""".strip()


def solicitar_analisis(
    clave: str,
    modelo: str,
    prompt: str,
    max_intentos: int = 4,
) -> dict:
    headers = {
        "Authorization": f"Bearer {clave}",
        "Content-Type": "application/json",
    }

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
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    for intento in range(1, max_intentos + 1):
        respuesta = requests.post(
            URL_API,
            headers=headers,
            json=datos,
            timeout=120,
        )

        if respuesta.status_code == 200:
            contenido = respuesta.json()["choices"][0]["message"]["content"]
            return json.loads(contenido)

        if respuesta.status_code == 429 and intento < max_intentos:
            espera = int(respuesta.headers.get("Retry-After", intento * 10))
            print(f"Límite temporal alcanzado. Esperamos {espera} segundos.")
            time.sleep(espera)
            continue

        mensaje = respuesta.text[:500]
        raise requests.HTTPError(
            f"Mistral devolvió {respuesta.status_code}: {mensaje}",
            response=respuesta,
        )

    raise RuntimeError("No se pudo obtener el análisis después de varios intentos.")


def validar_analisis(analisis: dict) -> dict:
    faltantes = [campo for campo in CAMPOS_ANALISIS if campo not in analisis]
    if faltantes:
        raise ValueError(f"Faltan campos en el análisis: {', '.join(faltantes)}")

    # Redondeamos las notas al medio punto más cercano.
    claridad = round(float(analisis["claridad_y_estructura"]) * 2) / 2
    agudeza = round(float(analisis["rigor_y_argumentacion"]) * 2) / 2

    analisis["claridad_y_estructura"] = max(1.0, min(10.0, claridad))
    analisis["rigor_y_argumentacion"] = max(1.0, min(10.0, agudeza))

    for campo in ["puntos_fuertes", "puntos_mejora"]:
        if not isinstance(analisis[campo], list):
            analisis[campo] = [str(analisis[campo])]

    return analisis


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

    claves = cargar_claves()
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
    print(f"Artículos pendientes: {len(pendientes)}")

    for numero, (_, articulo) in enumerate(pendientes.iterrows(), start=1):
        print(f"Analizando {numero}/{len(pendientes)}: {articulo['titulo']}")
        prompt = crear_prompt(articulo["titulo"], articulo["texto"])
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

        time.sleep(2)

    print(f"CSV generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_analisis()
