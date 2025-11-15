
Modelos v1: trataban de predecir merval_apertura (sin el shift por lo que en realidad esta mal) y usando el dataset limpio de variables economicas que armo Ian

Modelos v2: precedir preciio de apertura en t+2 usando las variables de dataset_v2_fs_top60_+2.csv.
            Es lo mismo que el modelo v4 de series temporales pero usando solo los 4 meses de text mining.
            Resultado: funciona bastante mal por problema de dimensionalidad (mismas features que filas)

Modelos v3: Mismo dataset que los modelos v2 pero agregando un feature selection para reducir problema de dimensionalidad
            Resultado: mejoran pero aun asi modelos lineales mas simples como el Ridge le ganan tanto al RF como al LGBM

Movelo v4: Seria comparable al modelo v2, no hacemos feature selection por lo que probablemente tenga problemas de dimensionalidad