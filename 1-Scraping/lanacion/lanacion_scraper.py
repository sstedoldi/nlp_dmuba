#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scraper de La Nación (Política/Economía) con rango de fechas y salida a Parquet/CSV.
Respetuoso de robots.txt: usa sitemaps y evita endpoints prohibidos.

Uso:
  python lanacion_scraper.py --start 2025-01-01 --end 2025-04-30 \
    --sections politica economia --out noticias_2025Q1.parquet --with-text

Requisitos:
  pip install requests beautifulsoup4 lxml pandas pyarrow (pyarrow opcional)
"""

import argparse
import datetime as dt
import gzip
import io
import json
import re
import sys
import time
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.lanacion.com.ar"
SITEMAP_INDEX_HIST = f"{BASE}/sitemap-index-historico.xml"
SITEMAP_INDEX = f"{BASE}/sitemap-index.xml"         # reciente
SITEMAP_NEWS = f"{BASE}/sitemap-news.xml"           # muy reciente
UA = "Mozilla/5.0 (compatible; SantiScraper/1.0; +https://example.com/bot)"


def yyyymm(date: dt.date) -> str:
    return f"{date.year:04d}-{date.month:02d}"


def http_get(url: str, sleep: float = 0.8, timeout: int = 20) -> requests.Response:
    """GET con UA y reintentos básicos."""
    for i in range(4):
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
            if r.status_code == 200:
                time.sleep(sleep)
                return r
            elif r.status_code in (403, 404):
                time.sleep(sleep)
                return r
        except requests.RequestException:
            time.sleep(1.5 * (i + 1))
    # última intención
    r = requests.get(url, headers={"User-Agent": UA}, timeout=timeout)
    time.sleep(sleep)
    return r


def parse_xml_bytes(data: bytes) -> BeautifulSoup:
    return BeautifulSoup(data, "xml")


def fetch_xml(url: str) -> Optional[BeautifulSoup]:
    r = http_get(url)
    if r.status_code != 200:
        return None
    content = r.content
    # algunos sitemaps pueden venir .gz aunque la URL sea .xml
    if r.headers.get("Content-Type", "").endswith("gzip") or url.endswith(".gz"):
        try:
            content = gzip.decompress(content)
        except OSError:
            pass
    return parse_xml_bytes(content)


def month_sitemaps_in_range(start: dt.date, end: dt.date) -> List[str]:
    """Toma los índices de sitemaps y retorna los sitemaps mensuales que intersectan el rango."""
    urls = set()

    # índice histórico (mensual/anual)
    doc_hist = fetch_xml(SITEMAP_INDEX_HIST)
    if doc_hist:
        for sm in doc_hist.find_all("sitemap"):
            loc = sm.loc.get_text(strip=True) if sm.loc else ""
            # heurística: incluyen "YYYY-MM" o similares
            if re.search(r"\d{4}-\d{2}", loc):
                m = re.search(r"(\d{4})-(\d{2})", loc)
                if m:
                    y, mo = int(m.group(1)), int(m.group(2))
                    first = dt.date(y, mo, 1)
                    last = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
                    # intersección con rango pedido
                    if not (last < start or first > end):
                        urls.add(loc)

    # índice “reciente”
    doc_idx = fetch_xml(SITEMAP_INDEX)
    if doc_idx:
        for sm in doc_idx.find_all("sitemap"):
            loc = sm.loc.get_text(strip=True) if sm.loc else ""
            urls.add(loc)

    # news (muy reciente, por las dudas)
    doc_news = fetch_xml(SITEMAP_NEWS)
    if doc_news:
        urls.add(SITEMAP_NEWS)

    return sorted(urls)


def extract_urls_from_sitemap(sm_url: str) -> List[Tuple[str, Optional[str]]]:
    """Devuelve [(url, lastmod_iso)] desde un sitemap o sitemap de noticias."""
    s = fetch_xml(sm_url)
    results = []
    if not s:
        return results

    # dos formatos comunes: <urlset><url> y <urlset><url><news:news>...
    for url_tag in s.find_all("url"):
        loc_tag = url_tag.find("loc")
        if not loc_tag:
            continue
        loc = loc_tag.get_text(strip=True)
        lastmod = None
        if url_tag.find("lastmod"):
            lastmod = url_tag.find("lastmod").get_text(strip=True)
        # intenta news:publication_date
        news_tag = url_tag.find(re.compile(r"news:news"))
        if news_tag:
            pub = news_tag.find(re.compile(r"news:publication_date"))
            if pub:
                lastmod = pub.get_text(strip=True)
        results.append((loc, lastmod))
    return results


def normalize_date(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    # admite 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS-03:00', etc.
    # strip microsegundos y TZ si hace falta
    try:
        return dt.datetime.fromisoformat(re.sub(r"Z$", "+00:00", s))
    except ValueError:
        # intenta solo fecha
        try:
            return dt.datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def url_matches_sections(url: str, wanted_sections: Iterable[str]) -> bool:
    # match por path: /politica/ o /economia/
    path = urlparse(url).path.lower()
    return any(f"/{sec.lower()}/" in path for sec in wanted_sections)


def extract_id_from_url(url: str) -> Optional[str]:
    # ejemplo: ...-nid07092025/  ó ...-nid18022025/
    m = re.search(r"-nid(\d+)", url)
    return m.group(1) if m else None


def parse_article(url: str, with_text: bool = False) -> dict:
    r = http_get(url, sleep=0.7)
    if r.status_code != 200:
        return {"url": url, "status": r.status_code}

    soup = BeautifulSoup(r.text, "lxml")

    # Título
    title = None
    if soup.title:
        title = soup.title.get_text(strip=True)
        # suele incluir " - LA NACION"
        title = re.sub(r"\s*-\s*LA NACION\s*$", "", title, flags=re.I)

    # Meta descripción (bajada)
    desc = None
    md = soup.find("meta", attrs={"name": "description"})
    if md and md.get("content"):
        desc = md["content"].strip()

    # Fecha y sección desde JSON-LD si existe
    pub_iso = None
    section = None
    for s in soup.find_all("script", type=re.compile("ld\\+json")):
        try:
            data = json.loads(s.string or "")
        except Exception:
            continue
        # puede venir como dict o lista de dicts
        items = data if isinstance(data, list) else [data]
        for obj in items:
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") in ("NewsArticle", "Article"):
                pub_iso = obj.get("datePublished") or obj.get("dateCreated") or pub_iso
                # a veces viene dentro de headline/section
                section = obj.get("articleSection") or section

    # fallback de fecha: buscar en meta property og:article:published_time
    if not pub_iso:
        meta_time = soup.find("meta", attrs={"property": "article:published_time"})
        if meta_time and meta_time.get("content"):
            pub_iso = meta_time["content"]

    # fallback de sección: del path
    if not section:
        section = urlparse(url).path.strip("/").split("/")[0] or None

    pub_dt = normalize_date(pub_iso)

    # Texto (opcional)
    cuerpo = None
    if with_text:
        # La nota suele tener párrafos en <p> dentro del cuerpo central.
        # Evitamos duplicar imágenes, figure, captions, etc.
        paragraphs = []
        for p in soup.find_all("p"):
            txt = p.get_text(" ", strip=True)
            if txt and len(txt) > 40:  # filtra menús, migas, etc.
                paragraphs.append(txt)
        if paragraphs:
            cuerpo = "\n\n".join(paragraphs)

    return {
        "id": extract_id_from_url(url),
        "url": url,
        "titulo": title,
        "bajada": desc,
        "seccion": section,
        "fecha_publicacion": pub_dt.isoformat() if pub_dt else None,
        "status": 200,
        "texto": cuerpo if with_text else None,
    }


def run(start: str, end: str, sections: List[str], out: str, with_text: bool):
    start_date = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end, "%Y-%m-%d").date()

    # Recolectar sitemaps relevantes
    sm_urls = month_sitemaps_in_range(start_date, end_date)
    if not sm_urls:
        print("No se encontraron sitemaps en el rango.", file=sys.stderr)
        sys.exit(2)

    print(f"Sitemaps a revisar ({len(sm_urls)}):")
    for u in sm_urls:
        print("  -", u)

    rows = []
    seen = set()

    for sm in sm_urls:
        print(f"[INFO] Leyendo sitemap: {sm}")
        try:
            pairs = extract_urls_from_sitemap(sm)
        except Exception as e:
            print(f"[WARN] Error leyendo {sm}: {e}", file=sys.stderr)
            continue

        for url, lastmod in pairs:
            if not url.startswith(BASE):
                continue
            if not url_matches_sections(url, sections):
                continue

            # filtro por fecha: usa lastmod si está, si no después parsea la nota
            ok_by_lastmod = False
            if lastmod:
                dtm = normalize_date(lastmod)
                if dtm and (start_date <= dtm.date() <= end_date):
                    ok_by_lastmod = True

            if not ok_by_lastmod:
                # todavía no sabemos: parsearemos la nota y filtramos por su fecha
                pass

            if url in seen:
                continue
            seen.add(url)

            art = parse_article(url, with_text=with_text)
            # filtro final por fecha_publicacion real
            pub = art.get("fecha_publicacion")
            keep = False
            if pub:
                try:
                    dpub = dt.datetime.fromisoformat(pub).date()
                    keep = (start_date <= dpub <= end_date)
                except Exception:
                    keep = False

            if keep:
                rows.append(art)

    if not rows:
        print("No se encontraron artículos en el rango/condiciones dadas.", file=sys.stderr)
        sys.exit(3)

    df = pd.DataFrame(rows)
    # columnas mínimas pedidas
    cols = ["id", "seccion", "fecha_publicacion", "titulo", "url"]
    extra = [c for c in ["bajada", "texto", "status"] if c in df.columns]
    df = df[cols + extra]

    # Guardado
    try:
        df.to_parquet(out, index=False)
        print(f"[OK] Guardado Parquet: {out} ({len(df)} filas)")
    except Exception as e:
        print(f"[WARN] Parquet falló ({e}). Guardando CSV…")
        out_csv = re.sub(r"\.parquet$", ".csv", out)
        df.to_csv(out_csv, index=False)
        print(f"[OK] Guardado CSV: {out_csv} ({len(df)} filas)")


def main():
    ap = argparse.ArgumentParser(description="Scraper La Nación (Política/Economía) con rango de fechas.")
    ap.add_argument("--start", required=True, help="Fecha inicio (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="Fecha fin (YYYY-MM-DD)")
    ap.add_argument("--sections", nargs="+", default=["politica", "economia"], help="Secciones a incluir")
    ap.add_argument("--out", required=True, help="Ruta de salida .parquet o .csv (si falla parquet)")
    ap.add_argument("--with-text", action="store_true", help="Incluir texto completo (más lento)")
    args = ap.parse_args()
    run(args.start, args.end, args.sections, args.out, args.with_text)


if __name__ == "__main__":
    main()
