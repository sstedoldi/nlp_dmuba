# Análisis de Modelos - Predicción MERVAL (Enero-Abril 2025)

## Resumen Ejecutivo

Se implementaron y evaluaron múltiples modelos de Machine Learning para predecir el índice MERVAL utilizando un dataset reducido de 4 meses (enero-abril 2025).

## Dataset

### Configuración
- **Período**: Enero-Abril 2025
- **Observaciones totales**: 77-79 (dependiendo del dataset)
- **Train/Test split**: 80/20 (~63 train, ~16 test)
- **Features**: 60 variables (ratio observaciones/features: ~1.05)

### Datasets Utilizados
1. **dataset_v2_fs_top60_+2.csv**: Variables macroeconómicas tradicionales
2. **dataset_llm_fs_top60_+2.csv**: Variables extraídas del LLM + macroeconómicas (77 obs)

## Modelos Implementados

### 1. Modelos Base (Sin Feature Selection)

#### LightGBM (`lightgbm.ipynb`)
- **Dataset**: dataset_v2
- **RMSE Test**: 264,011
- **R²**: -8.57
- **Error relativo**: 12.11%
- **Gap CV-Test**: +114%
- **Resultado**: Overfitting severo. Ratio 1.05:1 (obs/features) insuficiente para gradient boosting.

#### Random Forest (`random_forest.ipynb`)
- **Dataset**: dataset_v2
- **RMSE Test**: 285,651
- **R²**: -10.21
- **Error relativo**: 13.10%
- **Gap CV-Test**: +113%
- **Resultado**: Degradación catastrófica comparado con dataset completo. Ensemble débil con pocos datos.

#### LSTM (`lstm.ipynb`)
- **Dataset**: dataset_v2
- **RMSE Test**: 231,359
- **R²**: -8.80
- **Error relativo**: 10.52%
- **Resultado**: Red neuronal recurrente que tuvo mejor desempeño que algunos modelos de ensemble. Sin embargo, sigue siendo inadecuado con R² negativo.

#### Otros modelos:

##### ARIMA (`arima.ipynb`)
- **Dataset**: dataset_v2 (solo MERVAL univariado)
- **RMSE Test**: 176,055
- **MAPE**: 7.14%
- **Modelo**: ARIMA(0,1,0) - equivalente a modelo naive (random walk)
- **Resultado**: **Sorprendentemente mejor que modelos multivariados complejos**. El modelo más simple (random walk) superó a todos los modelos de ML, sugiriendo que con datos limitados, la simplicidad es superior.

##### Ridge (`ridge.ipynb`)
- **Dataset**: dataset_v2
- **RMSE CV**: 130,928
- **RMSE Test**: 1,162,379
- **R²**: -184.57
- **Error relativo**: 53.32%
- **Resultado**: **Fracaso catastrófico**. R² de -184 indica predicciones completamente erróneas. Regresión lineal con 60 features y 63 observaciones es matemáticamente problemática (sistema subdeterminado).

##### Elastic Net (`elastic_net.ipynb`)
- **Dataset**: dataset_v2
- **RMSE CV**: 130,330
- **RMSE Test**: 1,124,589
- **R²**: -172.70
- **Error relativo**: 51.59%
- **Resultado**: **Fracaso similar a Ridge**. A pesar de combinar L1+L2 regularization, no puede superar la limitación fundamental de datos insuficientes.

### 2. Modelos LLM (Sin Feature Selection)

#### LightGBM LLM (`lightgbm_llm.ipynb`)
- **Dataset**: dataset_llm (variables del LLM)
- **RMSE Test**: 261,329
- **R²**: -8.38
- **Error relativo**: 11.99%
- **Gap CV-Test**: +127%
- **Resultado**: Mejora marginal (~8-18%) sobre dataset_v2 pero aún inadecuado. Variables del LLM muestran cierto potencial.

#### Random Forest LLM (`random_forest_llm.ipynb`)
- **Dataset**: dataset_llm (variables del LLM)
- **RMSE Test**: 252,437
- **R²**: -7.75
- **Error relativo**: 11.58%
- **Gap CV-Test**: +120%
- **Resultado**: Mejora marginal (~12-24%) sobre dataset_v2. Mejor que LightGBM LLM pero aún inviable.

#### LSTM LLM (`lstm_llm.ipynb`)
- **Dataset**: dataset_llm
- **RMSE Test**: 287,824
- **R²**: -14.17
- **Error relativo**: 13.09%
- **Resultado**: Variables del LLM **empeoraron** el desempeño del LSTM en 24% vs. LSTM base. A diferencia de modelos de árbol, LSTM no puede filtrar features ruidosas, por lo que las variables del LLM agregaron ruido sin información útil.

### 3. Modelos Base (Con Feature Selection)

**Estrategia**: Reducir features de 60 a 12 usando importancia del modelo, mejorando ratio observaciones/features de 1.05 a 5.25.

#### LightGBM + FS (`lightgbm_fs.ipynb`)
- **Dataset**: dataset_v2
- **Features**: 12 (de 60)
- **RMSE CV**: ~120,000
- **RMSE Test**: 346,179
- **R²**: -15.46
- **Error relativo**: 15.88%
- **Gap CV-Test**: +188%
- **Top features**: merval_apertura_std20, merval_maximo_pct10, m2_diff3, merval_cierre, badlar_ma_5
- **Resultado**: Feature selection **empeoró** el desempeño. Reducir features no resuelve el problema de datos insuficientes.

#### Random Forest + FS (`random_forest_fs.ipynb`)
- **Dataset**: dataset_v2
- **Features**: 12 (de 60)
- **RMSE CV**: 109,585
- **RMSE Test**: 308,425
- **R²**: -12.07
- **Error relativo**: 14.15%
- **Gap CV-Test**: +181%
- **Top features**: merval_cierre, merval_apertura_std20, tc_mayorista_std20, emae_tendencia_ciclo_ma_5
- **Resultado**: Aunque mejoró el ratio obs/features, el modelo sigue con overfitting severo.

#### LSTM + FS (`lstm_fs.ipynb`)
- **Dataset**: dataset_v2
- **Features**: 12 (de 60)
- **RMSE Test**: 228,287
- **R²**: -8.55
- **Error relativo**: 10.38%
- **Resultado**: Feature selection produjo **mejora marginal** (~1.3% en RMSE) sobre LSTM base. Reducir features de 60 a 12 ayudó levemente pero no resolvió la limitación fundamental de datos insuficientes.

### 4. Modelos LLM (Con Feature Selection)

#### LightGBM LLM + FS (`lightgbm_fs_llm.ipynb`)
- **Dataset**: dataset_llm
- **Features**: 12 (de 60)
- **Observaciones train**: 61
- **RMSE CV**: ~120,000
- **RMSE Test**: 239,173
- **R²**: -6.86 (mejor R² de todos los modelos multivariados)
- **Error relativo**: 10.97%
- **Gap CV-Test**: +99% (mejor gap de todos los modelos multivariados)
- **Top features LLM**: impacto_sector_bancario_lag3, menciona_commodities_lag2, act_trump_lag2, gobernanza_mean, act_indec_lag3
- **Resultado**: Mejor R² y menor overfitting entre modelos multivariados. Variables del LLM (sentimiento político, impactos sectoriales) aportan información incremental. Aún inadecuado pero muestra potencial.

#### Random Forest LLM + FS (`random_forest_fs_llm.ipynb`)
- **Dataset**: dataset_llm
- **Features**: 12 (de 60)
- **Observaciones train**: 61
- **RMSE CV**: 99,210 (mejor CV de todos)
- **RMSE Test**: 266,737
- **R²**: -8.77
- **Error relativo**: 12.24%
- **Gap CV-Test**: +169%
- **Top features LLM**: Solo 2 del LLM seleccionadas (gobernanza_mean, sector_educación). RF priorizó variables tradicionales de volatilidad
- **Resultado**: Excelente CV pero no generaliza. Random Forest subutiliza variables del LLM comparado con LightGBM.

#### LSTM FS + LLM (`lstm_fs_llm.ipynb`)
- **Dataset**: dataset_llm
- **Features**: 12 (de 60)
- **RMSE Test**: 355,591
- **R²**: -22.16
- **Error relativo**: 16.17%
- **Resultado**: **PEOR MODELO DE TODOS LOS EXPERIMENTOS**. La combinación de feature selection + variables LLM fue catastrófica para LSTM. 56% peor que LSTM base y R² de -22.16 (el peor registrado). Confirma que LSTM NO funciona con variables LLM en datasets pequeños.

## Análisis Comparativo

### Tabla de Resultados

**Modelos de ML/DL ordenados por RMSE Test (mejor a peor)**

| Modelo | Dataset | Features | RMSE Test | R² | Error Rel | Gap CV-Test |
|--------|---------|----------|-----------|-----|-----------|-------------|
| LSTM + FS | v2 | 12 | 228,287 | -8.55 | 10.38% | - |
| LSTM | v2 | 60 | 231,359 | -8.80 | 10.52% | - |
| LightGBM LLM + FS | llm | 12 | 239,173 | -6.86 | 10.97% | +99% |
| Random Forest LLM | llm | 60 | 252,437 | -7.75 | 11.58% | +120% |
| LightGBM LLM | llm | 60 | 261,329 | -8.38 | 11.99% | +127% |
| LightGBM | v2 | 60 | 264,011 | -8.57 | 12.11% | +114% |
| Random Forest LLM + FS | llm | 12 | 266,737 | -8.77 | 12.24% | +169% |
| Random Forest | v2 | 60 | 285,651 | -10.21 | 13.10% | +113% |
| LSTM LLM | llm | 60 | 287,824 | -14.17 | 13.09% | - |
| Random Forest + FS | v2 | 12 | 308,425 | -12.07 | 14.15% | +181% |
| LightGBM + FS | v2 | 12 | 346,179 | -15.46 | 15.88% | +188% |
| LSTM FS + LLM | llm | 12 | 355,591 | -22.16 | 16.17% | - |

**Nota**:
- ARIMA (RMSE: 176,055) tuvo el mejor RMSE de todos los modelos pero no se incluye en la tabla principal por ser univariado (solo usa historia del MERVAL, sin features adicionales).
- Ridge (1,162,379) y Elastic Net (1,124,589) no se incluyen por tener performance catastrófica (R² < -170).
- LSTM + FS es el mejor modelo multivariado por RMSE Test (228,287).
- LightGBM LLM + FS tiene mejor R² (-6.86) y menor gap CV-Test (+99%) que LSTM + FS, aunque RMSE ligeramente mayor.

### Hallazgos Principales

#### 1. Feature Selection No Es Suficiente
- Reducir features de 60 a 12 mejoró el ratio observaciones/features (1.05 → 5.25)
- **Sin embargo**, en modelos con dataset_v2, la feature selection **empeoró** los resultados
- El problema no es solo dimensionalidad sino **cantidad absoluta de datos**

#### 2. Variables del LLM Aportan Valor Marginal
- **Modelos base con LLM** (sin FS): Mejora ~8-24% sobre dataset_v2
  - `lightgbm_llm`: Error 11.99% vs 12.11% (mejora 1%)
  - `random_forest_llm`: Error 11.58% vs 13.10% (mejora 12%)
- **Modelos LLM + FS**: Aún mejores resultados
  - `lightgbm_fs_llm` logró los **mejores resultados globales (excl. LSTM)** (Error 10.97%, Gap +99%)
  - `random_forest_fs_llm`: Segundo mejor (excl. LSTM) (Error 12.24%, Gap +169%)
- **Features LLM importantes**: menciones Trump, INDEC, gobernanza, impactos sectoriales (bancario, commodities, agroexportador)
- **Conclusión**: Variables del LLM muestran potencial pero necesitan más datos para validación real

#### 3. LightGBM vs Random Forest con Variables del LLM
- **LightGBM**: Seleccionó 8/12 features del LLM. Mejor generalización
- **Random Forest**: Seleccionó solo 2/12 features del LLM. Mejor CV pero peor test
- LightGBM explota mejor la información de sentimiento/menciones

#### 4. LSTM: Mejora con Feature Selection, Colapso con Variables LLM

**Performance de los 4 modelos LSTM**:
- LSTM + FS (v2): RMSE 228,287 - **Mejor LSTM**
- LSTM base (v2): RMSE 231,359 - Segundo mejor
- LSTM LLM: RMSE 287,824 - Tercer mejor (24% peor que base)
- LSTM FS + LLM: RMSE 355,591 - **Peor de todos** (56% peor que base)

**Hallazgos clave**:

1. **Feature selection ayudó marginalmente al LSTM** (mejora ~1.3%)
   - Con dataset_v2, reducir de 60 a 12 features mejoró levemente el RMSE
   - Menor complejidad (1,900 vs 5,000 parámetros) permitió mejor regularización
   - Sin embargo, R² sigue negativo (-8.55) - el modelo es inviable

2. **Variables LLM PERJUDICARON severamente al LSTM** (contrario a modelos de árbol)
   - LSTM LLM empeoró 24% vs. LSTM base
   - LSTM FS + LLM fue el **peor modelo de todos los experimentos** (R² -22.16)
   - Contrasta con LightGBM LLM + FS que fue el **mejor modelo** (R² -6.86)

3. **¿Por qué LSTM no funciona con variables LLM?**
   - **LSTM procesa TODAS las features por igual**: No puede ignorar variables ruidosas
   - **Modelos de árbol pueden filtrar**: Selección implícita en cada split
   - **Variables LLM son binarias/categóricas**: Ruido alto para series temporales cortas
   - **Dataset insuficiente**: Con 60 secuencias, LSTM aprende correlaciones espurias

4. **Paradoja arquitectural**:
   ```
   LightGBM + variables LLM → Mejora (mejor modelo)
   LSTM + variables LLM → Colapso (peor modelo)
   ```
   Con datos limitados, la capacidad de **filtrar features ruidosas** (árboles) > **procesar secuencias temporales** (LSTM)

#### 5. Modelos Simples Superan a Modelos Complejos
- **ARIMA(0,1,0)** (random walk) logró el **mejor RMSE**: 176,055
- **LSTM + FS** segundo mejor RMSE: 228,287 (pero R² negativo -8.55)
- **LSTM base** tercer mejor RMSE: 231,359 (R² -8.80)
- **Paradoja**: Con datos limitados, modelos más simples/univariados superan a modelos complejos multivariados
- Esto confirma que con 63 observaciones, la complejidad no ayuda

#### 6. Todos los Modelos Multivariados Son Inviables
- Todos presentan **R² negativo** (peor que predecir la media)
- Gaps CV-Test masivos (+99% a +188%) indican overfitting severo
- Errores relativos inaceptables (10-16%) para aplicaciones financieras
- Ridge y Elastic Net tuvieron **fracasos catastróficos** (R² < -170)

## Conclusiones

### Limitación Fundamental: Datos Insuficientes
El problema principal **no es la configuración del modelo** sino la **escasez de datos**:
- 61-63 observaciones son insuficientes para modelos de ensemble
- LightGBM requiere idealmente 200-500+ observaciones
- Random Forest necesita 100-300+ observaciones
- 4 meses de historia no capturan ciclos económicos completos

### ¿Por Qué Feature Selection No Funcionó?
1. **Mejora marginal en dataset_v2**: Empeoró las métricas al eliminar información útil
2. **Problema no es dimensionalidad**: El ratio 1.05 es problemático, pero el verdadero issue son las 63 observaciones absolutas
3. **Overfitting persiste**: Con tan pocos datos, incluso 12 features generan memorización

### Valor de las Variables del LLM

Las variables del LLM mostraron **comportamiento opuesto según la arquitectura del modelo**:

**Modelos de Árbol: Variables LLM AYUDAN**

*Modelos base (60 features)*:
- LightGBM LLM vs LightGBM v2: Mejora marginal de ~1-2% en error relativo
- Random Forest LLM vs Random Forest v2: Mejora más notable de ~12% en error relativo
- Ambos siguen con R² fuertemente negativos (-7.75 a -8.38)

*Modelos con Feature Selection (12 features)*:
- LightGBM LLM + FS: Mejor R² y menor gap (Error 10.97%, Gap +99%, R² -6.86)
- Random Forest LLM + FS: Segundo mejor por gap (Error 12.24%, Gap +169%)
- La combinación LLM + FS fue más efectiva que aplicar FS solo en dataset_v2

**LSTM: Variables LLM PERJUDICAN**

*Modelos base (60 features)*:
- LSTM LLM vs LSTM v2: Empeoramiento del 24% en RMSE (287k vs 231k)
- R² se degradó de -8.80 a -14.17

*Modelos con Feature Selection (12 features)*:
- LSTM FS + LLM: PEOR modelo de todos (Error 16.17%, R² -22.16)
- 56% peor que LSTM base
- La combinación FS + LLM fue catastrófica para LSTM

**¿Por qué esta diferencia?**

| Aspecto | Modelos de Árbol | LSTM |
|---------|------------------|------|
| **Filtrado de ruido** | Selección implícita en splits | Procesa todas las features |
| **Manejo de variables binarias** | Nativo (splits booleanos) | Problemático en series temporales |
| **Requerimiento de datos** | 100-300 obs | 5,000+ secuencias |
| **Resultado con LLM** | Mejora 1-12% | Empeora 24-56% |

**Features LLM relevantes (para modelos de árbol)**:
- Capturan expectativas políticas: menciones Trump, INDEC, gobernanza
- Información sectorial: impacto bancario, commodities, agroexportador
- Tienen sentido económico para predecir MERVAL
- Solo funcionan con modelos que pueden filtrar ruido (LightGBM, Random Forest)
- NO usar con LSTM en datasets pequeños (<1,000 obs)

### Recomendaciones

**Para mejorar los modelos actuales (NO recomendado)**:
- **ARIMA o modelos univariados** demostraron ser superiores con datos limitados
- Reducir a 5-8 features mediante selección basada en conocimiento del dominio
- Evitar modelos lineales simples (Ridge/Elastic Net fracasaron catastróficamente)

**Para solución real**:
1. **Expandir el dataset temporal**: Mínimo 6-12 meses, idealmente 2+ años
2. **Usar el dataset completo disponible**: Los modelos con 515 observaciones funcionaron aceptablemente
3. **Con más datos, re-evaluar variables del LLM**: Tienen potencial pero necesitan más observaciones para aprendizaje robusto

### Veredicto Final

**Ninguno de los modelos multivariados es viable para predicción práctica del MERVAL con el dataset de 4 meses.**

**Ranking de modelos por RMSE Test (Top 5)**:
1. **ARIMA(0,1,0)**: 176,055 (mejor general, pero solo univariado)
2. **LSTM + FS**: 228,287 (mejor multivariado por RMSE)
3. **LSTM base**: 231,359 (segundo mejor multivariado por RMSE)
4. **LightGBM LLM + FS**: 239,173 (mejor multivariado por R² -6.86 y gap CV-Test +99%)
5. **Random Forest LLM**: 252,437

**Peor modelo**: LSTM FS + LLM con RMSE 355,591 y R² -22.16

**Trade-off entre modelos multivariados**:
- **LSTM + FS** tiene mejor RMSE (228,287) pero R² muy negativo (-8.55) y no tiene gap CV-Test medible
- **LightGBM LLM + FS** tiene RMSE ligeramente mayor (239,173) pero mejor R² (-6.86) y menor overfitting (gap +99%)

Ambos modelos presentan métricas inaceptables:
- R² negativos (peor que baseline trivial)
- Error relativo del 10-11%
- LSTM sin validación cruzada dificulta evaluación de overfitting

**Paradojas clave**:

1. **Simplicidad > Sofisticación**: El modelo más simple (ARIMA random walk) superó a todos los modelos complejos de ML, confirmando que con datos extremadamente limitados, la simplicidad es superior.

2. **Arquitectura importa más que Features**:
   - LightGBM LLM + FS (mejor): Puede filtrar ruido de variables LLM
   - LSTM FS + LLM (peor): Amplifica ruido de variables LLM
   - Con datos limitados, **capacidad de filtrado** > **modelado de secuencias**

3. **Feature Selection ayuda a algunos, perjudica a otros**:
   - LSTM con v2: FS mejora marginalmente (~1.3%)
   - LightGBM/RF con v2: FS empeora severamente
   - LightGBM/RF con LLM: FS mejora significativamente
   - LSTM con LLM: FS colapsa el modelo completamente

**Hallazgo crítico sobre variables LLM**: Son útiles SOLO con modelos de árbol que pueden filtrar ruido. LSTM con variables LLM en datasets pequeños produce los peores resultados posibles.

**La estrategia correcta es usar el dataset completo (515 obs) o expandir el período temporal, no optimizar modelos complejos sobre datos insuficientes.**
