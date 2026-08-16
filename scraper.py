import json
import os
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


URL_AUTORA = "https://cronicavasca.elespanol.com/autor/jaione-sanz/"
ARCHIVO_SALIDA = "articulos_jaione_sanz.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/126 Safari/537.36"
    )
}


def descargar_pagina(url: str) -> BeautifulSoup:
    respuesta = requests.get(url, headers=HEADERS, timeout=20)
    respuesta.raise_for_status()
    return BeautifulSoup(respuesta.text, "html.parser")


def obtener_urls_articulos(max_paginas: int = 10) -> list[str]:
    urls = []

    for pagina in range(1, max_paginas + 1):
        url_pagina = URL_AUTORA if pagina == 1 else f"{URL_AUTORA}{pagina}/"
        print(f"Leyendo página {pagina}: {url_pagina}")

        try:
            soup = descargar_pagina(url_pagina)
        except requests.RequestException as error:
            print(f"No se pudo descargar la página: {error}")
            break

        # Buscamos los enlaces de los títulos publicados en la página.
        urls_pagina = []
        for enlace in soup.select("h2 a[href]"):
            url = urljoin(URL_AUTORA, enlace["href"])
            if "cronicavasca.elespanol.com" in url:
                urls_pagina.append(url)

        urls_nuevas = [url for url in urls_pagina if url not in urls]
        if not urls_nuevas:
            break

        urls.extend(urls_nuevas)
        time.sleep(1)

    return urls


def obtener_datos_json_ld(soup: BeautifulSoup) -> dict:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            datos = json.loads(script.get_text(strip=True))
        except (json.JSONDecodeError, TypeError):
            continue

        if isinstance(datos, dict):
            elementos = datos.get("@graph", [datos])
        elif isinstance(datos, list):
            elementos = datos
        else:
            continue

        for elemento in elementos:
            if not isinstance(elemento, dict):
                continue

            tipo = elemento.get("@type", "")
            if isinstance(tipo, list):
                es_articulo = any("Article" in valor for valor in tipo)
            else:
                es_articulo = "Article" in tipo

            if es_articulo:
                return elemento

    return {}


def extraer_articulo(url: str) -> dict:
    soup = descargar_pagina(url)
    datos_json = obtener_datos_json_ld(soup)

    titulo = datos_json.get("headline")
    if not titulo:
        etiqueta_titulo = soup.select_one("h1")
        titulo = (
            etiqueta_titulo.get_text(" ", strip=True)
            if etiqueta_titulo
            else "Sin título"
        )

    fecha = datos_json.get("datePublished", "")
    if not fecha:
        etiqueta_fecha = soup.select_one('meta[property="article:published_time"]')
        fecha = etiqueta_fecha.get("content", "") if etiqueta_fecha else ""

    resumen = datos_json.get("description", "")
    if not resumen:
        etiqueta_resumen = soup.select_one('meta[name="description"]')
        resumen = etiqueta_resumen.get("content", "") if etiqueta_resumen else ""

    texto = datos_json.get("articleBody", "")
    if not texto:
        parrafos = soup.select("article p") or soup.select("main p")
        texto = "\n".join(
            parrafo.get_text(" ", strip=True) for parrafo in parrafos
        )

    return {
        "titulo": titulo,
        "fecha": fecha,
        "url": url,
        "resumen_original": resumen,
        "texto": texto,
        "longitud_caracteres": len(texto),
    }


def cargar_articulos_anteriores() -> list[dict]:
    if not os.path.exists(ARCHIVO_SALIDA):
        return []

    df_anterior = pd.read_csv(ARCHIVO_SALIDA)
    return df_anterior.to_dict("records")


def guardar_articulos(articulos: list[dict]):
    df = pd.DataFrame(articulos)

    if not df.empty and "url" in df.columns:
        df = df.drop_duplicates(subset="url", keep="last")

    df.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")


def ejecutar_scraper():
    urls = obtener_urls_articulos()
    articulos = cargar_articulos_anteriores()
    urls_existentes = {articulo.get("url") for articulo in articulos}
    urls_pendientes = [url for url in urls if url not in urls_existentes]

    print(f"Artículos encontrados: {len(urls)}")
    print(f"Artículos ya guardados: {len(urls_existentes)}")
    print(f"Artículos nuevos: {len(urls_pendientes)}")

    for numero, url in enumerate(urls_pendientes, start=1):
        print(f"Descargando artículo nuevo {numero}/{len(urls_pendientes)}")

        try:
            articulos.append(extraer_articulo(url))
            guardar_articulos(articulos)
        except requests.RequestException as error:
            print(f"Error en {url}: {error}")

        # Esperamos para no sobrecargar el servidor.
        time.sleep(1)

    guardar_articulos(articulos)
    print(f"CSV generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    ejecutar_scraper()
