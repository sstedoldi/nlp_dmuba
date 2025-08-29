import requests
from bs4 import BeautifulSoup
import pandas as pd

import re

SECCION = "economia"


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


meses = {
    "enero": "01",
    "febrero": "02",
    "marzo": "03",
    "abril": "04",
    "mayo": "05",
    "junio": "06",
    "julio": "07",
    "agosto": "08",
    "septiembre": "09",
    "octubre": "10",
    "noviembre": "11",
    "diciembre": "12",
}


def parse_fecha(fecha_str):
    for mes, num in meses.items():
        if mes in fecha_str.lower():
            fecha_str = fecha_str.lower().replace(mes, num)
            break
    fecha_str = fecha_str.replace("de ", "").replace("-", "").strip()
    return pd.to_datetime(fecha_str, format="%d %m %Y %H:%M", errors="coerce")


def parse_fecha(fecha):
    if not isinstance(fecha, str):
        return fecha
    fecha_str = fecha.lower()
    for mes, num in meses.items():
        if mes in fecha_str:
            fecha_str = fecha_str.replace(mes, num)
            break
    fecha_str = re.sub(r"\bde\b", "", fecha_str)
    fecha_str = fecha_str.replace("-", "").strip()
    try:
        return pd.to_datetime(fecha_str, format="%d %m %Y %H:%M")
    except:
        return fecha


headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


try:
    df_url = pd.read_parquet("noticias_ambito.parquet", columns=["url"])
    urls_guardadas = set(df_url["url"].values)
except FileNotFoundError:
    df = pd.DataFrame(columns=["fecha", "titulo", "resumen", "articulo", "url"])
    df.to_parquet("noticias_ambito.parquet", index=False, engine="fastparquet")
    urls_guardadas = set()

for index in range(1, 1000):
    url = f"https://www.ambito.com/{SECCION}/{index}"

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

        if url_nota in urls_guardadas:
            print(f"La noticia {url_nota} ya fue procesada.")
            continue

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

        # Fecha
        fecha_tag = soup_nota.find("span", class_="news-headline__publication-date")
        fecha = fecha_tag.get_text(strip=True) if fecha_tag else "No encontrada"

        # Cuerpo del artículo
        summary_tag = soup_nota.find("h2", class_="news-headline__article-summary")
        summary = summary_tag.get_text() if summary_tag else "No encontrado"

        article_body_tag = None
        article_body_tags = soup_nota.find_all("article", class_="article-body")

        article_body = "No encontrado"
        if article_body_tags:
            article_body = " ".join([tag.get_text() for tag in article_body_tags])

        noticias.append(
            {
                "fecha": parse_fecha(fecha),
                "titulo": clean_text(titulo),
                "resumen": clean_text(summary),
                "articulo": clean_text(article_body),
                "url": url_nota,
                "seccion": SECCION,
            }
        )
        urls_guardadas.add(url_nota)
        print(f"Guardada la noticia: {url_nota}")

    if noticias:
        pd.DataFrame(noticias).to_parquet(
            "noticias_ambito.parquet",
            index=False,
            engine="fastparquet",
            append=True,
        )

        print("=" * 40)
        print(f"Página: {index}")
        print(f"Noticias nuevas: {len(noticias)}")
        print(f"Total acumulado: {len(urls_guardadas)}")
        print("=" * 40)
