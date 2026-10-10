"""Validación de limpieza_texto.py. Ejecutar con: python test_limpieza_texto.py"""

from bs4 import BeautifulSoup

from limpieza_texto import extraer_cuerpo, extraer_parrafos, limpiar_texto


def probar_cita_pegada():
    """Cita destacada sin punto final pegada al párrafo que la repite."""
    texto = (
        "Hasta que un día descubres que una cosa es tener opciones. "
        "Y aun así siempre habrá imbéciles en primetime, y por contagio en la "
        "barra del bar, responsabilizando al desgraciado de turno de que las "
        "cosas no le vayan mejor Y aun así siempre habrá imbéciles en "
        "primetime, y por contagio en la barra del bar, responsabilizando al "
        "desgraciado de turno de que las cosas no le vayan mejor. Supongo que "
        "es lo que tiene confundir buena suerte con talento."
    )
    limpio = limpiar_texto(texto)
    assert limpio.count("imbéciles") == 1, limpio
    assert limpio.endswith("con talento."), limpio
    assert limpio.startswith("Hasta que un día"), limpio


def probar_cita_con_punto():
    """La cita destacada aparece como oración suelta y se repite después."""
    texto = (
        "Primera idea del artículo, suficientemente larga para el umbral. "
        "La sombra dice más de la desigualdad ciudadana y nuestro consumista "
        "sistema que los informes. "
        "Hablando de desigualdad: La sombra dice más de la desigualdad "
        "ciudadana y nuestro consumista sistema que los informes. Cierre."
    )
    limpio = limpiar_texto(texto)
    assert limpio.count("consumista sistema") == 1, limpio
    assert limpio.endswith("Cierre."), limpio


def probar_texto_limpio_intacto():
    """Un texto sin duplicados no debe modificarse."""
    texto = (
        "Vas a morir. Con suerte, a los ochenta y tantos por pura "
        "obsolescencia. Antes, si se da la fatalidad de pasar bajo una "
        "maceta de geranios una tarde de viento. Eso dicta la lógica."
    )
    assert limpiar_texto(texto) == texto


def probar_texto_corto_no_tocado():
    """Las repeticiones de frases cortas legítimas no se eliminan."""
    texto = "Sí. Sí. No. Sí, claro. Repetir es natural en la conversación."
    assert limpiar_texto(texto) == texto


def probar_parrafos_excluye_citas():
    """Los párrafos de blockquote, pie de foto y comentarios no cuentan."""
    html = """
    <article><section class="article-body">
      <div class="article-body__content">
        <blockquote class="content__blockquote"><p>Cita destacada que no
        debe formar parte del cuerpo del artículo para nada.</p></blockquote>
        <p class="paragraph">Primer párrafo real del cuerpo del artículo.</p>
        <div class="article-media__caption"><p>Pie de foto de la imagen.</p></div>
        <p class="paragraph">Segundo párrafo real del cuerpo del artículo.</p>
        <p class="paragraph">Tercer párrafo real del cuerpo del artículo.</p>
      </div>
    </section>
    <div class="comment-text"><p>Comentario de lector.</p></div>
    </article>
    """
    parrafos = extraer_parrafos(BeautifulSoup(html, "html.parser"))
    assert len(parrafos) == 3, parrafos
    assert all("Cita destacada" not in p for p in parrafos)
    assert all("Pie de foto" not in p for p in parrafos)
    assert all("Comentario" not in p for p in parrafos)


def probar_cuerpo_prefiere_html():
    """Con HTML suficiente se ignora el articleBody con citas duplicadas."""
    html = """
    <article><div class="article-body__content">
      <p class="paragraph">Primer párrafo real, largo y detallado, del cuerpo.</p>
      <p class="paragraph">Segundo párrafo real, largo y detallado, del cuerpo.</p>
      <p class="paragraph">Tercer párrafo real, largo y detallado, del cuerpo.</p>
    </div></article>
    """
    article_body = (
        "Primer párrafo real, largo y detallado, del cuerpo. "
        "Cita destacada muy larga que solo está en el JSON-LD. "
        "Segundo párrafo real, largo y detallado, del cuerpo. "
        "Tercer párrafo real, largo y detallado, del cuerpo."
    )
    cuerpo = extraer_cuerpo(BeautifulSoup(html, "html.parser"), article_body)
    assert "Cita destacada" not in cuerpo
    assert cuerpo.count("Primer párrafo") == 1


def probar_cuerpo_respaldo_json_ld():
    """Sin HTML de cuerpo se usa el articleBody, pero limpio."""
    html = "<html><body><div class='otra-cosa'>x</div></body></html>"
    article_body = (
        "Texto inicial del artículo que viene del JSON-LD. "
        "Fragmento destacado inicial muy largo para superar el umbral y que "
        "se repite después. "
        "Fragmento destacado inicial muy largo para superar el umbral y que "
        "se repite después. Final."
    )
    cuerpo = extraer_cuerpo(BeautifulSoup(html, "html.parser"), article_body)
    assert cuerpo.count("Fragmento destacado") == 1, cuerpo


def probar_html_cronica_vasca_real():
    """Estructura real de Crónica Vasca con blockquotes intercalados (casos imbéciles y Confiésate)."""
    html = """
    <div class="article-body__content">
      <p class="paragraph" id="paragraph_5">Hasta que un día, oh, sorpresa, descubres que una cosa es tener opciones y otra muy diferente elegir.</p>
      <blockquote class="content__blockquote"><p>Y aun así siempre habrá imbéciles en <em>primetime</em>, y por contagio en la barra del bar, responsabilizando al desgraciado de turno de que las cosas no le vayan mejor</p></blockquote>
      <p class="paragraph" id="paragraph_6">Y aun así siempre habrá imbéciles en <em>primetime</em>, y por contagio en la barra del bar, responsabilizando al desgraciado de turno de que las cosas no le vayan mejor. Supongo que es lo que tiene confundir buena suerte, o una descomunal falta de escrúpulos, con talento y responsabilidad.</p>
      <p class="paragraph" id="paragraph_7">Maricarmen también podía elegir, decían.</p>
      <blockquote class="content__blockquote"><p>Esto es lo que le diría a quien se supone que he de votar en noviembre. Confiésate. Ponte de rodillas ante mí, siervo, y acepta lo que hiciste mal</p></blockquote>
      <p class="paragraph" id="paragraph_8">Esto es lo que le diría a quien se supone que he de votar en noviembre. Confiésate. Ponte de rodillas ante mí, siervo, y acepta lo que hiciste mal. Arrepiéntete. Y luego, explica cómo lo vas a corregir.</p>
    </div>
    """
    soup = BeautifulSoup(html, "html.parser")
    cuerpo = extraer_cuerpo(soup)
    assert cuerpo.count("imbéciles") == 1, cuerpo
    assert cuerpo.count("Confiésate") == 1, cuerpo
    assert "buena suerte" in cuerpo
    assert "Arrepiéntete" in cuerpo


def main():
    probar_cita_pegada()
    probar_cita_con_punto()
    probar_texto_limpio_intacto()
    probar_texto_corto_no_tocado()
    probar_parrafos_excluye_citas()
    probar_cuerpo_prefiere_html()
    probar_cuerpo_respaldo_json_ld()
    probar_html_cronica_vasca_real()
    print("OK: 8 pruebas superadas")


if __name__ == "__main__":
    main()
