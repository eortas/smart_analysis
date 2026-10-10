"""Extracción y limpieza del cuerpo real de los artículos.

La página de Crónica Vasca (y también su JSON-LD ``articleBody``) incluye los
fragmentos destacados (pull quotes) además de los párrafos del cuerpo, así que
el mismo texto aparece dos veces y el análisis detecta repeticiones falsas.

Este módulo resuelve el problema en dos niveles:
- ``extraer_parrafos``/``extraer_cuerpo``: selecciona solo los párrafos del
  cuerpo del artículo, excluyendo citas destacadas, pies de foto, sides y
  bloques de comentarios.
- ``limpiar_texto``: red de seguridad que elimina del texto cualquier
  fragmento largo que se repita a corta distancia (útil para los textos ya
  guardados en el CSV antes de corregir el scraper).
"""

import re
from bisect import bisect_left

# Un fragmento repetido debe medir al menos estos caracteres normalizados
# (sin espacios ni puntuación) para considerarse una cita duplicada.
MINIMO_DUPLICADO = 35

# Distancia máxima entre la primera copia y su repetición: las citas
# destacadas van pegadas al párrafo que las contiene, pero entre medias
# puede haber un párrafo entero, así que se da holgura. Con un mínimo de
# 35 caracteres repetidos exactamente, una repetición real de un texto
# propio sigue siendo prácticamente imposible fuera de las citas.
MARGEN_REPETICION = 6000

# Máximo de pasadas de limpieza por texto. Cada pasada elimina una copia
# suelta; una cita destacada larga puede estar partida en varias franjas,
# así que se permite holgura (el bucle se corta en cuanto no hay cambios).
MAX_PASADAS = 20

# Etiquetas cuyo contenido nunca forma parte del cuerpo del artículo.
BLOQUES_EXCLUIDOS = ("blockquote", "aside", "figure", "figcaption")

# Clases de ancestros o contenedores que marcan bloques ajenos al cuerpo del artículo.
CLASES_EXCLUIDAS = re.compile(
    r"caption|comment|sidebar|newsletter|related|advert|promo|social|share"
    r"|breadcrumb|most-read|quote|destacado|pullquote",
    re.IGNORECASE,
)


def _normalizar(texto: str) -> tuple[str, list[int]]:
    """Devuelve el texto sin espacios ni puntuación y el mapa de posiciones.

    ``mapa[i]`` es la posición en el texto original del i-ésimo carácter
    normalizado, lo que permite buscar por igualdad exacta (ignorando
    espacios y puntuación) y volver a cortar en el texto original.
    """
    caracteres = []
    mapa = []
    for posicion, caracter in enumerate(texto):
        if caracter.isalnum():
            caracteres.append(caracter.lower())
            mapa.append(posicion)
    return "".join(caracteres), mapa


def _franjas_oraciones(texto: str) -> list[tuple[int, int]]:
    """Divide el texto en fragmentos delimitados por finales de oración.

    Se corta con cualquier salto de espacio (incluidos saltos de línea y
    espacios no separables), no solo con espacios, porque en el HTML las
    citas destacadas van separadas del párrafo anterior por `\n\n`.
    """
    franjas = []
    inicio = 0
    for coincidencia in re.finditer(r"(?<=[.!?…])\s+", texto):
        franjas.append((inicio, coincidencia.start()))
        inicio = coincidencia.end()
    franjas.append((inicio, len(texto)))
    return franjas


def limpiar_texto(texto: str) -> str:
    """Elimina los fragmentos destacados repetidos y normaliza los espacios.

    Para cada fragmento del texto se comprueba si su comienzo (>=60
    caracteres normalizados) vuelve a aparecer poco después, ya sea dentro
    del propio fragmento o en los siguientes. Si es así, se conserva la
    última aparición —la que forma parte del cuerpo— y se elimina la
    primera, que es la cita destacada.
    """
    if not texto or not isinstance(texto, str):
        return texto

    texto = re.sub(r"[ \t]+", " ", texto).strip()

    for _ in range(MAX_PASADAS):
        normalizado, mapa = _normalizar(texto)
        eliminado = False

        for inicio, fin in _franjas_oraciones(texto):
            i = bisect_left(mapa, inicio)
            j = bisect_left(mapa, fin)
            if j - i < MINIMO_DUPLICADO:
                continue

            for largo in range(j - i, MINIMO_DUPLICADO - 1, -1):
                prefijo = normalizado[i : i + largo]
                repetido = normalizado.find(prefijo, i + 1)
                if repetido == -1:
                    continue

                origen_ini = mapa[i]
                origen_fin = mapa[i + largo - 1] + 1

                # La copia posterior debe empezar tras la primera (nada de
                # autocoincidencias dentro de la propia copia) y no muy
                # lejos: solo las citas destacadas van pegadas al texto.
                if mapa[repetido] < origen_fin:
                    continue
                if mapa[repetido] - origen_fin > MARGEN_REPETICION:
                    continue

                texto = texto[:origen_ini] + texto[origen_fin:]
                eliminado = True
                break

            if eliminado:
                break

        texto = re.sub(r" {2,}", " ", texto).strip()
        if not eliminado:
            break

    return texto


def extraer_parrafos(soup) -> list[str]:
    """Devuelve solo los párrafos del cuerpo del artículo.

    Prefiere el contenedor del cuerpo (``div.article-body__content``); si no
    existe, usa ``article``/``main``. En cualquier caso descarta los párrafos
    dentro de citas destacadas, asides, figuras o bloques ajenos al texto.
    """
    contenedor = soup.select_one("div.article-body__content")
    if contenedor is None:
        contenedor = soup.select_one("article") or soup.select_one("main")
    if contenedor is None:
        return []

    # En Crónica Vasca / El Español los párrafos del cuerpo usan la clase "paragraph",
    # mientras que las citas destacadas van en blockquote.content__blockquote.
    parrafos_candidatos = contenedor.select("p.paragraph")
    if len(parrafos_candidatos) < 3:
        parrafos_candidatos = contenedor.select("p")

    parrafos = []
    for parrafo in parrafos_candidatos:
        if parrafo.find_parent(BLOQUES_EXCLUIDOS) is not None:
            continue
        if parrafo.find_parent(class_=CLASES_EXCLUIDAS) is not None:
            continue
        clases_p = parrafo.get("class") or []
        if any(CLASES_EXCLUIDAS.search(c) for c in clases_p):
            continue
        texto = parrafo.get_text(" ", strip=True)
        if texto:
            parrafos.append(texto)

    return parrafos


def extraer_cuerpo(soup, article_body: str = "") -> str:
    """Obtiene el cuerpo limpio del artículo.

    Usa el HTML cuando ofrece párrafos suficientes (es la fuente más fiable,
    ya que las citas destacadas quedan fuera del cuerpo) y el
    ``articleBody`` del JSON-LD como respaldo; en ambos casos aplica
    ``limpiar_texto`` por si quedara alguna repetición.
    """
    parrafos = extraer_parrafos(soup)
    texto_html = "\n\n".join(parrafos)

    if len(parrafos) >= 3 or not (article_body or "").strip():
        return limpiar_texto(texto_html)

    return limpiar_texto(article_body)

