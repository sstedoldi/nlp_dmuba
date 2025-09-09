from pathlib import Path
import pandas as pd
pd.set_option("display.max_colwidth", 200)
pd.set_option("display.width", 140)

DATA_PATH = Path("./1-Scraping/lanacion/noticias_LN_2025Q1.parquet")

if DATA_PATH.suffix.lower() == ".parquet":
    df = pd.read_parquet(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

df.to_csv("noticias_LN_2025Q1.csv", index=False)

print("Dimensiones:", df.shape)
print(df.head(1).to_markdown())