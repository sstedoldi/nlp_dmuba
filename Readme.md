Perfecto 🚀 Te propongo una versión mejorada y más ordenada del **README.md**, con un **Quick Start** al inicio (para que cualquiera pueda correr el proyecto rápido), una narrativa más clara y un formato profesional.

---

````markdown
# Proyecto Final de NLP / Text Mining  
**Maestría en Data Mining – UBA**

---

## Tema: LLM-based Feature Engineering

Este trabajo práctico final explora el uso de **Large Language Models (LLMs)** como herramienta para la generación de *features* a partir de textos periodísticos. El objetivo es evaluar cómo estas representaciones enriquecidas impactan en modelos predictivos tradicionales, con énfasis en el **forecasting de variables macroeconómicas** (ej. EMAE, inflación).  

---

## 🚀 Quick Start

1. Clonar el repositorio:
   ```bash
   git clone <URL_DEL_REPO>
   cd <NOMBRE_DEL_REPO>
````

2. Crear entorno virtual en **Python 3.10**:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. Instalar dependencias:

   ```bash
   pip install -r requirements.txt
   ```
   
---

## 🎯 Objetivos

* Diseñar un **pipeline completo** para la recolección y preprocesamiento de noticias de medios digitales.
* **Periodo de análisis:** 2025-01 a 2025-04.
* **Fuentes:** Ámbito (Tomás), Clarín (Ian), Página/12 (Euge), Infobae (Leo), La Nación (Santi), Argentina.gob.
* **Campos a scrapear:** `{Fecha, Título, Sección, Diario, Contenido}`.
* Aplicar LLMs para extraer *features* de alto nivel (**sentimiento, embeddings, clasificaciones contextuales**).
* Integrar dichas *features* en modelos de predicción y compararlas frente a un baseline sin LLMs.
* Evaluar **eficiencia y trade-offs** entre LLMs en la nube (GPT-4o mini / GPT-5 nano) y modelos locales (\~3B).
* Analizar **escalabilidad mediante procesamiento asincrónico**.

---

## 🛠️ Metodología

1. **Definición de la fuente de datos**

   * Selección de diarios nacionales.
   * Determinación de secciones relevantes (economía y política).
   * Delimitación del período temporal.

2. **Ingesta y construcción del dataset**

   * Scraping automatizado para obtener el *top-10* de noticias diarias por fuente.
   * Limpieza, normalización y almacenamiento en formato estructurado (Parquet).
   * Dataset base con variables `{fuente, sección, fecha, texto}`.

   Recursos:

   * [RSS La Nación](https://servicios.lanacion.com.ar/herramientas/rss/ayuda)
   * [Scraping ejemplos](https://hernanescu.github.io/2019/01/scraping-media/)

3. **LLM-based Feature Engineering**

   * **Prompt Engineering:** diseño de instrucciones para sentimiento y relevancia.
   * **Clasificación:** outputs binarios y escalas (1 a 10).
   * **Embeddings contextuales:** representaciones vectoriales con LLMs.
   * **Procesamiento asincrónico:** reducción de tiempo/costo con *requests* paralelos.

4. **Modelado predictivo**

   * Entrenamiento de modelos de forecasting económico (Regresión, ARIMA, XGBoost).
   * Comparación entre:

     * Baseline (sin features de LLMs)
     * Modelo + Features LLM
     * LLM predict directo
   * Evaluación de impacto en métricas predictivas.

---

## 📌 Etapas de Desarrollo

* **A. Fuente de datos** → definición de diarios, secciones y periodo.
* **B. Pipeline de ingesta** → scraping, limpieza y dataset.
* **C. Feature Engineering con LLMs** → prompts, embeddings, señales semánticas.
* **D. Procesamiento asincrónico** → requests paralelos a APIs/modelos locales.
* **E. Modelado** → baseline vs modelos con features de LLM.
* **F. Evaluación comparativa** → resultados, impacto y trade-offs.

---

## 👥 Integrantes – Grupo 4

* Eugenio Negrin
* Ian Link
* Santiago Tedoldi
* Tomás Elsesser
* Leopoldo Serpa

---

## 📅 Hitos

* **10/09/2025** → Entrega de **PPT + Dataset**.

---

## ⚙️ Tecnologías y Herramientas

* **Lenguaje:** Python 3.10
* **Librerías:** Scrapy, Pandas, Scikit-learn, AsyncIO, Transformers
* **LLMs:** GPT-4o mini, GPT-5 nano, modelos locales (\~3B)
* **Infraestructura:** GPU Laptop + APIs OpenAI

---

## 📊 Resultados esperados

* Dataset curado de noticias con *features* derivadas de LLMs.
* Evaluación cuantitativa del valor agregado en modelos macroeconómicos.
* Benchmark detallado entre **LLMs en la nube** y **modelos locales**.
* Documento final con conclusiones metodológicas y técnicas sobre **LLM-based Feature Engineering**.

```

---

¿Querés que también arme un **diagrama simple en Markdown/mermaid** con el pipeline (Scraping → Preprocesamiento → LLM Features → Modelado → Evaluación)?
```



