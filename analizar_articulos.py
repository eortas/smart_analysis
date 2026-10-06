import json
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


ARCHIVO_ENTRADA = "articulos_jaione_sanz.csv"
ARCHIVO_SALIDA = "evaluacion_articulos_jaione_sanz.csv"
URL_API = "https://api.mistral.ai/v1/chat/completions"
VERSION_CRITERIO = "opinion_v4"

MODELO = "ministral-8b-latest"

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


def crear_prompt(titulo: str, texto: str) -> str:
    return f"""
Analiza el siguiente artículo de opinión en español.

Valora el artículo como un editor experto en periodismo y en comunicación.
Sé estrictamente profesional y valora los artículos de opinión con ese rigor.

Devuelve solamente un objeto JSON con esta estructura exacta:
{{
  "resumen_tematico": "Resumen neutral de 2 o 3 frases",
  "estilo_predominante": "Una categoría breve",
  "critica_editorial": "2 o 3 frases con los fallos más importantes del texto",
  "claridad_y_estructura": 6.5,
  "rigor_y_argumentacion": 7.5,
  "puntos_fuertes": ["Punto concreto", "Punto concreto"],
  "puntos_mejora": ["Punto concreto", "Punto concreto"]
}}

Las puntuaciones deben estar entre 1 y 10 y utilizar incrementos de 0.5.
Usa activamente los medios puntos cuando la valoración quede entre dos niveles.
Evalúa únicamente el contenido recibido y no inventes datos externos.
Incluye entre 2 y 4 elementos en cada lista.

Antes de puntuar, escribe "critica_editorial" con los fallos más importantes
que encuentres en el texto. Si no encuentras fallos, relee: siempre los hay.
Deja que esa crítica condicione después las puntuaciones.

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

Calibración de notas (escala de 1 a 10, siempre en saltos de 0.5).
Como editor experto que juzga una colección de unas 90 columnas publicadas,
esta es la distribución que cabe esperar; no todas las piezas merecen nota de
publicación impecable:
- 9.5-10 (unas 3 o 4 de cada 90): pieza excepcional, brillante, casi sin
  debilidades. Regálala solo a lo verdaderamente extraordinario.
- 8.5-9 (unas 15 de cada 90): pieza muy buena, con virtudes claras y fallos
  menores.
- 7.5-8 (unas 27 de cada 90): pieza sólida y correcta, con limitaciones
  puntuales.
- 6-7 (unas 22 de cada 90): pieza irregular o plana; se lee sin disgustar,
  pero no deja huella.
- 4.5-5.5 (unas 13 de cada 90): pieza floja, tesis débil, tópicos o desarrollo
  perezoso.
- 1-4 (unas 8 de cada 90): suspenso claro: incoherente, superficial o
  directamente insostenible.

Reglas duras:
- Nunca otorgues más de 8 sin poder citar al menos dos virtudes excepcionales
  y específicas del propio texto.
- Si el artículo carece de tesis clara, se apoya en tópicos de manual o repite
  ideas, su nota máxima es 6.5.
- Si además es superficial, incoherente o poco convincente, suspéndelo (5 o
  menos): no tengas reparo en dar suspenso a quien no cumple su nivel.
- Tampoco seas rancano: si un artículo es excepcional, puntúalo con 9.5 o 10.
- Si todas tus notas acaban entre 8 y 9, no estás ejerciendo de editor.

Ejemplos de calibración (compara el artículo que recibes con estos casos):
- "Otra vez los políticos prometen lo mismo. Es una vergüenza cómo nos toman
  por tontos. Los ciudadanos merecemos respeto." -> claridad 4.5, rigor 4.0
  (lugar común puro, sin tesis ni desarrollo).
- "Dicen que el tiempo lo cura todo, pero nadie explica por qué el reloj de
  la sala de espera va más lento. Corremos tanto que olvidamos para qué." ->
  claridad 6.5, rigor 6.5 (arranque prometedor, desarrollo deslabazado).
- "El 'sí pero no' del Gobierno ante la vivienda revela su prioridad real: el
  suelo no es hogar, es activo. Mientras expropiar suene a palabra tabú, el
  alquiler seguirá siendo una carrera de obstáculos." ->
  claridad 8.0, rigor 8.5 (tesis clara, ilustración eficaz, ahonda poco).
- Pieza con tesis original, ejemplos concretos, cierre redondo, ironía
  precisa y sin un solo párrafo muerto -> claridad 9.5, rigor 9.5.

Una publicación profesional no merece automáticamente una nota alta. Penaliza
los tópicos, la reiteración, la falta de una tesis clara, los saltos lógicos,
la ironía forzada, la crítica superficial y los finales poco efectivos.

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
        "temperature": 0.9,
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

        time.sleep(4)

    print(f"CSV generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_analisis()
