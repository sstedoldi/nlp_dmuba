#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from dateutil import parser as dateparser
from collections import OrderedDict
import pytz
import feedparser
import pandas as pd
import sys
import time

LOCAL_TZ = pytz.timezone("America/Argentina/Buenos_Aires")

# --- Configura aquí tus diarios y sus RSS principales ---
# Puedes agregar o quitar feeds. Se toman hasta 10 por diario (combinando sus feeds).
DIARIOS_RSS = OrderedDict({
    "La Nación": [
        "https://www.lanacion.com.ar/rss/tema/ultimas-noticias.xml",
        "https://www.lanacion.com.ar/rss/canalpolitica.xml",
        "https://www.lanacion.com.ar/rss/canaleconomia.xml",
    ],
    "Clarín": [
        "https://www.clarin.com/rss/lo-ultimo/",
        "https://www.clarin.com/rss/politica/",
        "https://www.clarin.com/rss/economia/",
    ],
    "Página/12": [
        "https://www.pagina12.com.ar/rss/portada",
        "https://www.pagina12.com.ar/rss/secciones/el-pais",
        "https://www.pagina12.com.ar/rss/secciones/economia",
    ],
    "Infobae": [
        "https://www.infobae.com/feeds/rss/",
        "https://www.infobae.com/america/rss.xml",
    ],
    "Ámbito": [
        "https://www.ambito.com/rss/ultimas-noticias.xml",
        "https://www.ambito.com/rss/economia.xml",
    ],
})

def normalize_dt(dt_str):
    """Parsea fechas RSS a timezone local (Buenos Aires) y devuelve naive ISO date (YYYY-MM-DD) + datetime completo."""
    if not dt_str:
        # si no hay fecha, usamos ahora
        dt = datetime.now(tz=LOCAL_TZ)
    else:
        try:
            dt = dateparser.parse(dt_str)
            # si no tiene tz, asumimos UTC (muchos RSS publican en GMT)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            dt = dt.astimezone(LOCAL_TZ)
        except Exception:
            dt = datetime.now(tz=LOCAL_TZ)
    day_str = dt.strftime("%Y-%m-%d")
    return day_str, dt

def get_top10_for_source(source_name, rss_list, per_source_limit=10, request_pause=0.6):
    """Lee múltiples feeds de un diario, mezcla y ordena por fecha desc, de-duplica títulos y retorna top N."""
    rows = []
    seen_titles = set()

    for rss_url in rss_list:
        try:
            d = feedparser.parse(rss_url)
        except Exception as e:
            print(f"[WARN] No se pudo leer {rss_url}: {e}", file=sys.stderr)
            continue

        for entry in d.entries:
            title = (entry.title or "").strip()
            link = getattr(entry, "link", "").strip()
            # published/parsing robusto
            published = getattr(entry, "published", None) or getattr(entry, "updated", None)
            day_str, dt_local = normalize_dt(published)

            # De-duplicación simple por título (puedes cambiar a hash(title+domain) si hiciera falta)
            key = title.lower()
            if not title or key in seen_titles:
                continue
            seen_titles.add(key)

            rows.append({
                "date": day_str,                           # YYYY-MM-DD en Buenos Aires
                "published_at_local": dt_local.isoformat(),# Fecha-hora local
                "source": source_name,
                "title": title,
                "link": link
            })

        # evitar ser agresivos con los servidores (aunque RSS es liviano)
        time.sleep(request_pause)

    # Ordena por fecha-hora local (desc) y corta top N
    rows.sort(key=lambda r: r["published_at_local"], reverse=True)
    rows = rows[:per_source_limit]

    # Añade ranking dentro del diario
    for i, r in enumerate(rows, start=1):
        r["rank_in_source"] = i

    return rows

def main():
    parser = argparse.ArgumentParser(description="Scraping (vía RSS) del top 10 de noticias por diario.")
    parser.add_argument("--out", default="noticias_top10.csv", help="Ruta de salida (CSV o .parquet)")
    parser.add_argument("--sources", nargs="*", default=[],
                        help="Filtrar por nombre de diario (tal cual figura en la config). Ej: --sources 'La Nación' Clarín")
    parser.add_argument("--limit", type=int, default=10, help="Top N por diario (default 10)")
    parser.add_argument("--pause", type=float, default=0.6, help="Pausa (seg) entre requests a feeds")
    args = parser.parse_args()

    selected = DIARIOS_RSS
    if args.sources:
        selected = OrderedDict((k, v) for k, v in DIARIOS_RSS.items() if k in set(args.sources))
        if not selected:
            print("[ERROR] No se encontraron diarios con esos nombres. Revisa --sources.", file=sys.stderr)
            sys.exit(1)

    all_rows = []
    for source, feeds in selected.items():
        print(f"[INFO] Procesando {source}...")
        rows = get_top10_for_source(source, feeds, per_source_limit=args.limit, request_pause=args.pause)
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No se obtuvieron noticias. ¿Feeds caídos o filtros muy restrictivos?", file=sys.stderr)

    df = pd.DataFrame(all_rows, columns=[
        "date", "published_at_local", "source", "rank_in_source", "title", "link"
    ])

    # Guardado
    out_path = args.out
    if out_path.lower().endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[OK] Guardado: {out_path}")
    # Muestra una vista rápida
    with pd.option_context("display.max_colwidth", 100):
        print(df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
