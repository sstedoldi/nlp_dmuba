
1. lightgbm, random_forest, lstm: 
    dataset_v2_fs_top60_+2.csv
    Sin feature selection adicional
    Mismos modelos que los mejores de ST pero ajustados para tragbajar con 4 meses en lugar de + de 2 años

2. lightgbm_llm, random_forest, lstm_llm:
    dataset_llm_fs_top60_+2.csv
    Sin feature selection adicional
    Mismos modelos que los de 1. pero cambiando el dataset
    No se hizo ningun ajuste adicional ya que los 2 datasets tienen el mismo periodo y la misma cantidad de features.

3. lightgbm_fs, random_forest_fs, lstm_fs:
    dataset_v2_fs_top60_+2.csv
    Con feature selection (12-15 features) para evitar maldicion de la dimensionalidad
    Mismos modelos que en 1. pero agregando paso de feature selection antes de la optimizacion

4. lightgbm_fs_llm, random_forest_fs_llm, lstm_fs_llm:
    dataset_llm_fs_top60_+2.csv
    Con feature selection (12-15 features) para evitar maldicion de la dimensionalidad
    Mismos modelos que en 2. pero agregando paso de feature selection antes de la optimizacion

