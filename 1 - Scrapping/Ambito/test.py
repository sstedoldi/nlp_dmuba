import requests
from bs4 import BeautifulSoup
import pandas as pd


# URL de ejemplo – sección económica

headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
df = pd.DataFrame(columns=["fecha", "titulo", "resumen", "articulo", "url"])
df.to_csv("noticias_ambito.csv", index=False, encoding="utf-8-sig")

for index in range(1, 50):
    url = f"https://www.ambito.com/economia/{index}"

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error {response.status_code} al solicitar la página.")

    soup = BeautifulSoup(response.text, "html.parser")
    articulos = soup.find_all("article", class_="news-article")

    noticias = []

    for art in articulos:
        a_tag = art.find("a", href=True)
        if not a_tag:
            continue

        url_nota = a_tag["href"]

        # 2. Entrar a cada nota
        resp_nota = requests.get(url_nota, headers=headers)
        print(resp_nota)
        if resp_nota.status_code != 200:
            continue

        soup_nota = BeautifulSoup(resp_nota.text, "html.parser")

        # 3. Extraer datos
        # Título
        titulo_tag = soup_nota.find("h1", class_="news-headline__title")
        titulo = titulo_tag.get_text(strip=True) if titulo_tag else "No encontrado"
        titulo = titulo.replace('"', "")

        # Fecha
        fecha_tag = soup_nota.find("span", class_="news-headline__publication-date")
        fecha = fecha_tag.get_text(strip=True) if fecha_tag else "No encontrada"

        # Cuerpo del artículo
        summary_tag = soup_nota.find("h2", class_="news-headline__article-summary")
        summary = summary_tag.get_text() if summary_tag else "No encontrado"
        summary = summary.replace('"', "")

        article_body_tag = None
        article_body_tags = soup_nota.find_all("article", class_="article-body")

        article_body = "No encontrado"
        if article_body_tags:
            article_body = " ".join([tag.get_text() for tag in article_body_tags])

        noticias.append(
            {
                "fecha": fecha,
                "titulo": titulo,
                "resumen": summary,
                "articulo": article_body,
                "url": url_nota,
            }
        )

    # 4. Guardar en tabla
    df = pd.DataFrame(noticias)
    df.to_csv(
        "noticias_ambito.csv", mode="a", header=False, index=False, encoding="utf-8-sig"
    )


# Opcional: exportar a CSV
