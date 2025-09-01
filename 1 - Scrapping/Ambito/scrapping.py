import asyncio
import aiohttp
from bs4 import BeautifulSoup
import pandas as pd
import re
import shutil
from datetime import datetime
from pathlib import Path

SECCION = "politica"


# --- Helpers ---
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
        print(f"Error al parsear la fecha: {fecha}")
        return fecha


headers = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}

# --- Archivos ---
script_dir = Path(__file__).parent
backup_dir = script_dir / "backups"
backup_dir.mkdir(exist_ok=True)

parquet_file = script_dir / "noticias_ambito.parquet"

if parquet_file.exists():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = backup_dir / f"noticias_ambito_backup_{timestamp}.parquet"
    shutil.copy(parquet_file, backup_file)
    print(f"Backup creado en: {backup_file}")

try:
    df_url = pd.read_parquet(parquet_file, columns=["url"])
    urls_guardadas = set(df_url["url"].values)
except FileNotFoundError:
    df = pd.DataFrame(columns=["fecha", "titulo", "resumen", "articulo", "url"])
    df.to_parquet(parquet_file, index=False, engine="fastparquet")
    urls_guardadas = set()


# --- Función async para scrapear una noticia ---
async def fetch_noticia(session, url_nota):
    try:
        async with session.get(url_nota, headers=headers) as resp:
            if resp.status != 200:
                return None

            html = await resp.text()
            soup_nota = BeautifulSoup(html, "html.parser")

            # Saltar notas en vivo
            if soup_nota.find("span", class_="news-headline-lbp__live-badge"):
                return None

            titulo_tag = soup_nota.find("h1", class_="news-headline__title")
            titulo = titulo_tag.get_text(strip=True) if titulo_tag else "No encontrado"

            fecha_tag = soup_nota.find("span", class_="news-headline__publication-date")
            fecha = fecha_tag.get_text(strip=True) if fecha_tag else "No encontrada"

            summary_tag = soup_nota.find("h2", class_="news-headline__article-summary")
            summary = summary_tag.get_text() if summary_tag else "No encontrado"

            article_body_tags = soup_nota.find_all("article", class_="article-body")
            article_body = (
                " ".join(tag.get_text() for tag in article_body_tags)
                if article_body_tags
                else "No encontrado"
            )

            return {
                "fecha": parse_fecha(fecha),
                "titulo": clean_text(titulo),
                "resumen": clean_text(summary),
                "articulo": clean_text(article_body),
                "url": url_nota,
                "seccion": SECCION,
            }
    except Exception as e:
        print(f"Error en {url_nota}: {e}")
        return None


# --- Main ---
async def main():
    async with aiohttp.ClientSession() as session:
        for index in range(300, 1000):
            url = f"https://www.ambito.com/{SECCION}/{index}"
            resp = await session.get(url, headers=headers)
            if resp.status != 200:
                print(f"Error {resp.status} en la página {url}")
                continue

            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            articulos = soup.find_all("article", class_="news-article")

            # Juntar todas las URLs nuevas
            urls_nuevas = []
            for art in articulos:
                a_tag = art.find("a", href=True)
                if not a_tag:
                    continue
                url_nota = a_tag["href"]
                if url_nota not in urls_guardadas:
                    urls_nuevas.append(url_nota)
                    urls_guardadas.add(url_nota)

            # Pedir todas las noticias en paralelo
            tasks = [fetch_noticia(session, url) for url in urls_nuevas]
            resultados = await asyncio.gather(*tasks)

            noticias = [r for r in resultados if r is not None]
            if noticias:
                pd.DataFrame(noticias).to_parquet(
                    parquet_file,
                    index=False,
                    engine="fastparquet",
                    append=True,
                )
                print(
                    f"Página {index} -> {len(noticias)} noticias nuevas - Fecha: {noticias[0]['fecha']} - Total: {len(urls_guardadas)}"
                )
                if noticias[0]["fecha"] < pd.to_datetime("2025-01-01"):
                    print("Noticias anteriores a 2025, deteniendo el scrapping.")
                    break
            else:
                print(f"Página {index} -> 0 noticias nuevas")


if __name__ == "__main__":
    asyncio.run(main())
