# Proyecto Final de NLP / Text Mining  
**Maestría en Data Mining – UBA**

---

## Tema: LLM-based Feature Engineering

Este trabajo práctico final se centra en explorar el uso de **Large Language Models (LLMs)** como herramienta para la generación de *features* a partir de textos. El objetivo es evaluar cómo estas representaciones enriquecidas impactan en modelos predictivos tradicionales, con énfasis en forecasting de variables macroeconómicas (ej. EMAE, inflación).  

---

## Objetivos

- Diseñar un pipeline completo para la recolección y preprocesamiento de noticias de medios digitales.
- Periodo de tiempo: 2025-01 a 2025-04
- Fuentes: ambito (Tomas), clarin (Ian), pagina12 (Euge), infobae (Leo), lanacion (Santi), argentina.gob
- Campos a scrapper: {Fecha, Titulo, Seccion, Diario, Contenido}  
- Aplicar LLMs para extraer *features* de alto nivel (sentimiento, embeddings, clasificaciones contextuales).  
- Integrar dichas *features* en modelos de predicción y compararlas frente a un baseline sin información de LLMs.  
- Evaluar la eficiencia y trade-offs entre el uso de **LLMs en la nube (GPT-4o mini / GPT-5 nano)**.  
- Analizar la escalabilidad mediante procesamiento asincrónico.  

---

## Metodología

1. **Definición de la fuente de datos**  
   - Selección de diarios nacionales e internacionales.  
   - Determinación de secciones relevantes (principalmente economía y política).  
   - Delimitación del período temporal a analizar.  

2. **Ingesta y construcción del dataset**  
   - Implementación de scraping automatizado para obtener el top-10 de noticias diarias por fuente.  
   - Limpieza de texto, normalización y almacenamiento estructurado.  
   - Creación de un dataset con variables base (fuente, sección, fecha, texto). 

   Links útiles:
   https://servicios.lanacion.com.ar/herramientas/rss/ayuda
   https://hernanescu.github.io/2019/01/scraping-media/

3. **LLM-based Feature Engineering**  
   - **Prompt Engineering:** diseño de instrucciones para evaluar sentimiento y relevancia.  
   - **Clasificación y escalado:** outputs binarios (positivo/negativo) y escalas (1 a 10).  
   - **Embeddings contextuales:** generación de representaciones vectoriales con LLMs.  
   - **Procesamiento asincrónico:** optimización de requests para reducir tiempo y costo.  

4. **Modelado predictivo**  
   - Entrenamiento de modelos de forecasting económico (ej. regresión, ARIMA, XGBoost).  
   - Comparación con un baseline sin features de LLMs.
   - Comparación de la predicciones: Modelo baseline vs. Modelo + LLMs features vs. LLM predict  
   - Evaluación de impacto en métricas predictivas.  

---

## Etapas de Desarrollo

- **A. Fuente de datos:** definición de diarios, secciones y periodo de análisis.  
- **B. Pipeline de ingesta:** scraping, limpieza y creación del dataset.  
- **C. LLM-based Feature Engineering:** prompts, embeddings y extracción de señales semánticas.  
- **D. Procesamiento asincrónico:** requests paralelos a APIs / modelos locales.  
- **E. Modelado:** baseline vs modelos con features de LLM.  
- **F. Evaluación comparativa:** resultados, impacto y trade-offs.  

---

## Integrantes del Grupo 4

- Eugenio Negrin  
- Ian Link  
- Santiago Tedoldi  
- Tomás Elsesser  
- Leopoldo Serpa  

---

## Tecnologías y Herramientas

- **Lenguajes:** Python  
- **Librerías:** Scrapy, Pandas, Scikit-learn, AsyncIO, Transformers  
- **LLMs:** GPT-4o mini, GPT-5 nano, modelos locales (~3B)  
- **Infraestructura:** GPU Laptop, APIs OpenAI  

---

## Resultados esperados

- Dataset curado de noticias con *features* derivadas de LLMs.  
- Evaluación cuantitativa del valor agregado de dichas *features* en modelos macroeconómicos.  
- Benchmark detallado entre **LLMs cloud** y **modelos locales**.  
- Documento final con conclusiones metodológicas y técnicas sobre **LLM-based feature engineering**.  



