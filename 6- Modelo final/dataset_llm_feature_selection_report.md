
# Feature Selection Report
Fecha: 2025-11-13 20:14:35

- Input: `dataset_pre_modelo.csv`
- Filtrado C1 (missing > 1.00%): 8
- Filtrado C2 (dominancia >= 99.00%): 22
- Clusters C3 (|corr| >= 95%): 1153
- Mandatorias retenidas: 0
- Top-20 por método: XGB=30, Boruta=30, SHAP=30
- Top-60 final: 60

Parámetros:
- N_SPLITS=5
- TARGET=merval_apertura_+2
