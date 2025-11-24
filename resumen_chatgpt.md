# Guía de repaso – Modelos de lenguaje, embeddings, pretraining y alineamiento

> Basada en las clases de la materia, en *Speech and Language Processing* de Jurafsky & Martin (3rd ed. draft) [1] y en papers clásicos de la literatura de NLP [2–8].

---

## 1. ¿Qué es transfer learning?

El *transfer learning* es un paradigma de aprendizaje donde un modelo se entrena primero en una tarea o dominio fuente (típicamente grande y general, como un corpus masivo de texto) y luego se reutiliza parcial o totalmente para una tarea objetivo distinta, usualmente con menos datos etiquetados. La idea central es que el modelo aprende representaciones generales (p. ej., del lenguaje) que luego se “transfieren” a nuevas tareas mediante *fine-tuning* o capas adicionales específicas de tarea [1,10].

En NLP moderno, el pretraining de grandes modelos de lenguaje sobre texto masivo sin etiquetas (LM, MLM, etc.) seguido de fine-tuning supervisado en tareas específicas (clasificación, NER, QA, etc.) es el caso paradigmático de *transfer learning* [1,10].

---

## 2. ¿Qué es la semántica distribucional?

La semántica distribucional se basa en la hipótesis distribucional: “palabras que aparecen en contextos similares tienden a tener significados similares” [1,3,6]. Formalmente, se construye un espacio vectorial donde cada palabra está representada por un vector que resume sus patrones de coocurrencia con otras palabras (en ventanas, documentos, etc.).

Modelos clásicos: matrices palabra×contexto con pesos como conteos, TF–IDF o PPMI, y métodos de reducción de dimensionalidad (SVD, etc.). Modelos neuronales modernos (word2vec, GloVe, fastText) implementan esta idea con objetivos de predicción (skip-gram, CBOW) [3,6].

---

## 3. ¿Por qué son importantes los embeddings?

Los *embeddings* (vectores densos de baja dimensión) son importantes porque:

1. Transforman tokens discretos (palabras, subpalabras) en vectores reales que pueden ser procesados por redes neuronales (diferenciables).
2. Capturan información semántica y sintáctica: similitud de significados se refleja como proximidad geométrica (coseno alto, etc.) [1,3].
3. Reducen dimensionalidad respecto de representaciones dispersas (one-hot, bolsas de palabras).
4. Permiten compartir información entre palabras similares; esto mejora la generalización, especialmente con datos escasos.
5. Son reutilizables: embeddings pre-entrenados pueden usarse en múltiples modelos y tareas (transfer learning).

---

## 4. ¿Qué es un embedding?

Un *embedding* es una función de mapeo de un espacio discreto (p. ej. el vocabulario de tokens) a un espacio vectorial continuo de baja dimensión, típicamente $$\mathbb{R}^d$$. Cada token $$w$$ se representa como un vector $$\mathbf{v}_w \in \mathbb{R}^d$$.

En modelos neuronales, el embedding se implementa como una tabla de parámetros aprendibles $$E \in \mathbb{R}^{|V|\times d}$$; el índice del token selecciona una fila de esa tabla [1,7]. Estos vectores pueden ser estáticos (una sola representación por palabra) o contextuales (varían según la oración).

---

## 5. ¿Qué son embeddings contextuales?

Los *embeddings contextuales* son vectores que representan un token en función de su contexto específico en la secuencia. La misma palabra en diferentes oraciones tendrá embeddings distintos, dependiendo de las otras palabras y la estructura de la frase [1,11].

Se obtienen con modelos como ELMo, BERT o GPT: la secuencia completa se procesa con RNNs o Transformers; en cada capa se produce un vector por posición que incorpora información del contexto cercano (y, en modelos bidireccionales, de ambos lados). Estos embeddings resuelven limitaciones de los embeddings estáticos para polisemia y desambiguación de sentido.

---

## 6. ¿Cómo se implementa el transfer learning en una red neuronal?

En práctica, el *transfer learning* se implementa típicamente así:

1. **Pretraining**: Entrenar una red grande (p. ej., un Transformer) en una tarea auto-supervisada general (LM, MLM, NSP, etc.) usando enorme cantidad de texto no etiquetado [1,10].
2. **Inicialización**: Para la tarea objetivo, se inicializan los parámetros del modelo con los pesos pre-entrenados.
3. **Adaptación** (*fine-tuning* o similares):

   * Añadir una cabeza de tarea (p. ej., una capa de clasificación sobre el embedding [CLS] de BERT).
   * Entrenar el modelo en el dataset etiquetado de la tarea, actualizando todos los parámetros (fine-tuning completo) o solo una parte (cabeza, adapters, LoRA, etc.).
4. **Opcional**: Congelar capas inferiores y entrenar solo las superiores o módulos adicionales para evitar *overfitting* y reducir costo computacional.

---

## 7. ¿Cuáles son las ventajas del transfer learning en términos de los datos de entrenamiento?

Principales ventajas:

1. **Menor requerimiento de datos etiquetados**: El modelo ya ha aprendido gramática, semántica básica y conocimiento de mundo durante el pretraining. El dataset de la tarea puede ser relativamente pequeño [1,10].
2. **Mejor generalización**: El modelo parte de representaciones ricas y robustas, menos propensas a sobreajuste en tareas específicas.
3. **Aprovechamiento de datos no etiquetados**: El pretraining usa grandes corpus sin anotación humana, mucho más abundantes y baratos.
4. **Reutilización multi-tarea y multi-dominio**: Un mismo modelo pre-entrenado puede adaptarse a distintas tareas y dominios con pocas muestras cada uno.

---

## 8. ¿Qué interpretación tiene que dos palabras se encuentren “cerca” en un espacio vectorial?

Si medimos la similitud entre vectores (típicamente coseno), que dos palabras estén “cerca” significa que sus patrones de uso en el corpus son similares. Esto refleja que:

* Aparecen en contextos similares (vecinos, documentos, etc.).
* Tienen roles sintácticos y semánticos parecidos (p. ej., *gato* y *perro*).

En semántica distribucional, cercanía geométrica ≈ similitud semántica/funcional. Sin embargo, también puede capturar asociaciones (p. ej., *doctor* y *hospital*) y puede heredar sesgos de los datos [1,3,6].

---

## 9. Describa el método Skip-Gram en word2vec

En Skip-Gram (Mikolov et al. 2013) [2], el objetivo es aprender embeddings de palabras maximizando la probabilidad de los contextos dado la palabra central. Dado un corpus tokenizado, para cada posición $$t$$ con palabra $$w_t$$, se consideran palabras en una ventana $$(t-c,\dots,t+c)$$ como contextos $$w_{t+j}$$.

Objetivo básico (simplificado) para un par (palabra central $$w$$, palabra contexto $$c$$):

$$
\max_{\theta} \sum_{(w,c)} \log p(c \mid w; \theta)
$$

Donde $$p(c \mid w)$$ se define con una capa softmax sobre el producto escalar de los vectores de palabra y de contexto. En práctica se usa *negative sampling* o *hierarchical softmax* para eficiencia. El entrenamiento produce dos matrices de embeddings; suele usarse la de palabras o la combinación de ambas [2,3].

---

## 10. Describa el método GloVe

GloVe (Global Vectors) [4] es un modelo de embeddings que utiliza conteos de coocurrencia globales (matriz palabra×contexto). Para cada par (palabra $$i$$, contexto $$j$$) con conteo $$X_{ij}$$, el modelo aprende vectores $$\mathbf{w}_i$$, $$\tilde{\mathbf{w}}_j$$ y sesgos $$b_i, \tilde{b}_j$$ minimizando:

$$
J = \sum_{i,j} f(X_{ij})\big(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}*j - \log X*{ij}\big)^2
$$

donde $$f(\cdot)$$ es una función de ponderación que reduce el peso de coocurrencias muy raras o muy frecuentes. Intuición: los productos escalares de los vectores deben aproximar los log-conteos de coocurrencia. GloVe combina ventajas de modelos basados en conteos y basados en predicción [4].

---

## 11. ¿Cuál es la diferencia conceptual entre word2vec y GloVe?

* **Word2vec (Skip-Gram/CBOW)**:

  * Modelo de predicción local; entrena con ejemplos (palabra centro, contexto) extraídos de ventanas de contexto.
  * Parámetros se optimizan para predecir contextos a partir de palabras (o viceversa) sin usar explícitamente toda la matriz global de coocurrencias.
* **GloVe**:

  * Modelo basado en conteos globales; utiliza la matriz de coocurrencia palabra×contexto.
  * Objetivo explícito: factorizar log-conteos para capturar relaciones de razón de coocurrencias.

En la práctica, ambos producen embeddings similares en calidad, pero GloVe aprovecha información global explícita, mientras word2vec lo hace implícitamente vía entrenamiento local [3,4].

---

## 12. ¿Qué es la tokenización?

La tokenización es el proceso de segmentar una secuencia de caracteres (texto bruto) en unidades mínimas procesables llamadas *tokens*: palabras, subpalabras o caracteres [1,2].

En NLP clásico, tokenizar implica separar palabras, puntuación, números, abreviaturas (p. ej., estándar Penn Treebank). En modelos modernos, se suelen usar tokenizadores de subpalabras como BPE o unigram LM, que construyen un vocabulario de unidades (palabras frecuentes, raíces, sufijos) y segmentan texto en esas unidades para manejar vocabularios abiertos y reducir problemas de *out-of-vocabulary* [1].

---

## 13. ¿Qué es un modelo de lenguaje?

Un modelo de lenguaje (LM) es un modelo probabilístico que asigna una probabilidad a secuencias de tokens. Idealmente estima:

$$
p(w_1, \dots, w_T) = \prod_{t=1}^T p(w_t \mid w_1, \dots, w_{t-1})
$$

En su forma clásica, un LM n-gram asume una dependencia Markov de orden $$n-1$$:

$$
p(w_t \mid w_1,\dots,w_{t-1}) \approx p(w_t \mid w_{t-n+1}, \dots, w_{t-1})
$$

En versiones neuronales (RNN, LSTM, Transformer decoder), la historia previa se codifica en un estado latente o en una representación contextual, pero el objetivo sigue siendo modelar la probabilidad de la secuencia y, a partir de ello, generar texto o servir como componente en tareas de NLP [1,3,7,8].

---

## 14. ¿Qué es un modelo de lenguaje unidireccional?

Es un LM que condiciona cada token solo en el *pasado* (tokens anteriores), nunca en el futuro. Formalmente: $$p(w_t \mid w_1,\dots,w_{t-1})$$.

Ejemplos:

* Modelos n-gram clásicos.
* RNN/LSTM entrenados como LM izquierda→derecha.
* Transformers decoder-only tipo GPT [8].

Esta restricción unidireccional es esencial para generación autoregresiva y para tareas donde no se conoce el futuro (traducción autoregresiva, generación de texto).

---

## 15. ¿Qué es un modelo de lenguaje generativo?

Es un modelo de lenguaje que no solo asigna probabilidades a secuencias, sino que puede *muestrear* nuevas secuencias, generando texto token a token conforme a la distribución aprendida. Generalmente se entrena autoregresivamente (next-token prediction) y en inferencia se usa búsqueda (greedy, beam search, nucleus sampling, etc.) para producir texto coherente.

Los LLM tipo GPT son modelos de lenguaje generativos; también lo son n-grams si se usan para generar texto, aunque su capacidad es mucho más limitada [1,8].

---

## 16. Explique el proceso autoregresivo de los modelos generativos

En un modelo autoregresivo, la generación ocurre iterativamente:

1. Se tiene un *prompt* o contexto inicial $$w_1,\dots,w_k$$.
2. El modelo estima la distribución $$p(w_{k+1} \mid w_1,\dots,w_k)$$.
3. Se elige un token según una estrategia (argmax, muestreo, top-k, nucleus).
4. El token elegido se concatena al contexto, y se repite el proceso para $$w_{k+2}$$, etc.

Formalmente,

$$
p(w_1, \dots, w_T) = \prod_{t=1}^T p(w_t \mid w_{<t})
$$

Este esquema es la base de la generación en GPT, LLaMA y otros modelos decoder-only [8,10].

---

## 17. ¿Qué es un modelo de lenguaje n-gram?

Es un LM que aproxima la probabilidad de la siguiente palabra usando solo las últimas $$n-1$$ palabras como contexto. Por ejemplo, un trigram ($$n=3$$) usa $$p(w_t \mid w_{t-2}, w_{t-1})$$. La probabilidad se estima mediante conteos en un corpus y se suaviza (Laplace, Kneser–Ney, etc.) para lidiar con n-grams raros o no observados [1,3].

---

## 18. ¿Cómo se entrena un modelo n-gram?

Pasos típicos [1,3]:

1. Tokenizar el corpus y definir vocabulario (incluyendo token UNK).

2. Contar frecuencias: $$C(w_{t-n+1},\dots,w_{t-1}, w_t)$$.

3. Estimar probabilidades máximoverosímiles:

   $$
   \hat{p}(w_t \mid w_{t-n+1}^{t-1}) = \frac{C(w_{t-n+1}^{t})}{C(w_{t-n+1}^{t-1})}
   $$

4. Aplicar *smoothing*: Laplace, Good–Turing, Kneser–Ney, interpolación/backoff para reasignar probabilidad a eventos no vistos.

5. Evaluar en un conjunto de prueba con *perplexity*.

---

## 19. ¿Cuáles son las principales desventajas de un modelo n-gram?

1. **Contexto corto**: solo mira $$n-1$$ palabras, incapaz de capturar dependencias largas.
2. **Explosión combinatoria**: número de n-grams crece exponencialmente con $$n$$; la mayoría son raros o no aparecen.
3. **Datos escasos**: incluso con corpus grandes, muchos n-grams plausibles no se observan; requiere smoothing complejo.
4. **Memoria y almacenamiento**: tablas de n-grams grandes para vocabularios amplios.
5. **Poca capacidad de generalización semántica**: trata palabras como símbolos discretos sin compartir información entre similares [1,3].

---

## 20. ¿Cuál es la importancia del vocabulario en un modelo de lenguaje?

El vocabulario define el conjunto de tokens que el modelo puede manejar directamente. Aspectos clave [1]:

* **Cobertura**: un vocabulario pequeño reduce *OOV* pero obliga a dividir palabras en subpalabras; uno enorme complica entrenamiento y memoria.
* **Granularidad**: palabras vs subpalabras vs caracteres; afecta capacidad de modelar morfología, neologismos y nombres propios.
* **Distribución de frecuencia**: palabras raras pueden ser agrupadas en UNK o representadas por combinaciones de subpalabras.

Diseñar el vocabulario (p. ej., mediante BPE con tamaño ~30k–100k tokens) es crucial para equilibrio entre eficiencia y capacidad de representación.

---

## 21. ¿Qué es la lematización?

La lematización es el proceso de mapear formas flexionadas o derivadas de una palabra a su forma canónica o *lema* (p. ej. *canté, cantaba, cantarán* → *cantar*). A diferencia del stemming, la lematización usa información morfológica y, a veces, sintáctica para devolver una forma léxica válida [1].

---

## 22. ¿Qué ventajas trae la lematización?

Ventajas:

1. **Normalización semántica**: agrupa variantes morfológicas bajo un mismo lema, reduciendo sparsity.
2. **Mejora en recuperación de información y matching**: una consulta *correr* puede recuperar documentos con *corriendo*, *corrió*, etc.
3. **Mejoras en tareas de análisis**: en análisis temático o tópicos, contar lemas en lugar de formas aisladas produce estadísticas más robustas.
4. **Particularmente útil en lenguajes con morfología rica** (eslavo, romance, etc.).

---

## 23. ¿Cuándo conviene lematizar y cuándo no? ¿Para qué tareas?

Conviene lematizar cuando:

* Interesa el contenido semántico más que la forma morfológica exacta (clasificación de documentos, IR, topic modeling, algunos modelos basados en bolsa de palabras).
* El idioma tiene flexión rica que dispersa frecuencias en muchas formas.

No conviene (o se usa con cuidado) cuando:

* La morfología es informativa para la tarea (análisis de sentimientos sutil, NER, etiquetado morfosintáctico, QA fina).
* Se usan modelos basados en embeddings contextuales (BERT, etc.), que ya manejan bien variación morfológica; la lematización puede eliminar matices de significado.

---

## 24. ¿Qué es Precision, Recall y F1?

En un problema de clasificación binaria (positivo/negativo), dado: TP (verdaderos positivos), FP (falsos positivos), FN (falsos negativos):

* **Precision**: proporción de predicciones positivas que son correctas.

  $$
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  $$

* **Recall** (sensibilidad): proporción de ejemplos positivos reales que se recuperan.

  $$
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  $$

* **F1**: media armónica entre precision y recall.

  $$
  F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
  $$

En clasificación multi-clase o secuencial (NER), se usan macro/micro promedios [1].

---

## 25. Ejemplos donde conviene maximizar Precision vs Recall

* **Maximizar Precision**:

  * Detección de fraude en tarjetas donde las revisiones manuales son costosas; se prefiere que cuando el modelo alerte, normalmente sea fraude real.
  * Clasificación de correos como “spam” cuando falsos positivos son muy dañinos (pérdida de emails importantes).

* **Maximizar Recall**:

  * Detección temprana de enfermedades graves (screening de cáncer); se prioriza no perder casos positivos aunque haya falsos positivos.
  * NER para anonimización de datos sensibles (identificar personas, direcciones); es preferible marcar de más antes que dejar datos sin anonimizar.

---

## 26. ¿Cómo se evalúan los modelos de lenguaje?

Métricas típicas [1,3,10]:

1. **Perplexity**: mide cuán “sorprendido” está el modelo por un corpus de prueba.
2. **Log-likelihood promedio** o *cross-entropy*: equivalente logarítmico de la perplexity.
3. **Tareas extrínsecas**: impacto en downstream tasks (traducción, QA, clasificación).
4. **Benchmarks estandarizados**: GLUE, SuperGLUE, MMLU, etc. para LLMs [10].
5. **Evaluaciones humanas**: coherencia, fluidez, veracidad, utilidad.

---

## 27. ¿Qué es perplexity?

Dado un conjunto de prueba $$w_1,\dots,w_T$$, la log-verosimilitud media del modelo es:

$$
\ell = -\frac{1}{T} \sum_{t=1}^T \log_2 p(w_t \mid w_{<t})
$$

La *perplexity* se define como:

$$
\text{PP} = 2^{\ell} = 2^{-\frac{1}{T} \sum_t \log_2 p(w_t \mid w_{<t})}
$$

Intuitivamente es el inverso de la probabilidad geométrica media de las palabras: un modelo “menos perplejo” (PP baja) asigna mayor probabilidad a los datos reales. Es análogo al tamaño efectivo del conjunto de palabras plausibles en cada paso [1,3].

---

## 28. ¿Qué relación hay entre word embeddings y redes neuronales? ¿Por qué maridan tan bien?

Los embeddings suelen ser la primera capa de una red neuronal para NLP. Relación clave:

* La red ve como entrada vectores densos (embeddings) en lugar de one-hot; esto reduce dimensionalidad y permite representar similitud semántica.
* Los embeddings se aprenden conjuntamente con el resto de la red (o se inicializan con vectores preentrenados).
* Capas superiores (RNN, CNN, Transformer) operan sobre estos vectores, componiendo significado para frases y documentos.

“Maridan bien” porque los embeddings transforman el problema de símbolos discretos en uno de geometría en $$\mathbb{R}^d$$, que es el terreno natural del aprendizaje profundo [1,7].

---

## 29. Diseñe un modelo de clasificación de texto con una red neuronal feed-forward

Esquema básico (bolsa de palabras + FFNN):

1. **Representación de entrada**:

   * Tokenizar el texto.
   * Mapear cada token a un embedding $$\mathbf{v}_i$$.
   * Agregar los vectores (suma o promedio) para obtener una representación del documento $$\mathbf{h}_0$$.

2. **Red feed-forward**:

   $$
   \mathbf{h}_1 = \sigma(W_1 \mathbf{h}_0 + \mathbf{b}_1)
   $$

   $$
   \mathbf{o} = W_2 \mathbf{h}_1 + \mathbf{b}_2
   $$

   donde $$\sigma$$ es no linealidad (ReLU, tanh).

3. **Salida**:

   * Softmax sobre $$\mathbf{o}$$ para obtener distribución sobre clases (p. ej., *positivo/negativo*).

4. **Entrenamiento**:

   * Minimizar cross-entropy entre predicción y etiqueta usando SGD/Adam.

Esto puede extenderse con varias capas ocultas, regularización (dropout) y embeddings pre-entrenados.

---

## 30. ¿Qué características suelen tener los métodos de tokenización? ¿Cuál es el objetivo?

Objetivo general: segmentar texto en unidades que equilibren:

* Cobertura (manejar vocabulario abierto).
* Eficiencia (tamaño de vocabulario razonable).
* Capacidad semántica y morfológica.

Características comunes de métodos modernos (BPE, unigram LM) [1]:

* Aprenden un vocabulario de subpalabras a partir de un corpus grande.
* Prefieren unidades frecuentes (palabras completas, raíces) y descomponen palabras raras.
* Garantizan descomposición de cualquier secuencia de bytes/caracteres.
* Son deterministas una vez entrenados (mismo texto → mismos tokens).

---

## 31. ¿Qué son redes neuronales recurrentes (RNN)?

Son arquitecturas neuronales diseñadas para secuencias, donde el modelo mantiene un estado oculto que se actualiza recurrentemente al procesar cada elemento de entrada [1,8]:

$$
\mathbf{h}_t = f(W_x \mathbf{x}*t + W_h \mathbf{h}*{t-1} + \mathbf{b})
$$

El estado $$\mathbf{h}_t$$ resume información de la historia pasada; se usa para predicción (p. ej., LM) o para producir una representación de toda la secuencia. Ejemplos: RNN simple, LSTM, GRU [1,8].

---

## 32. Describa una célula recurrente. ¿Qué elementos la componen?

Para una RNN simple:

* Entrada en el tiempo t: $$\mathbf{x}_t$$ (embedding de palabra).
* Estado oculto previo: $$\mathbf{h}_{t-1}$$.
* Transformaciones lineales: matrices $$W_x, W_h$$.
* No linealidad: $$f$$ (tanh, ReLU).
* Nuevo estado: $$\mathbf{h}_t = f(W_x \mathbf{x}*t + W_h \mathbf{h}*{t-1} + \mathbf{b})$$.

En LSTM/GRU, la célula incluye puertas (gates) que controlan qué información se olvida, se actualiza o se expone, resolviendo problemas de gradientes a largo plazo [1,8].

---

## 33. ¿Qué ventajas tienen las redes recurrentes respecto al modelo n-gram en términos del largo de la secuencia?

* Pueden, en principio, modelar dependencias de largo alcance: el estado $$\mathbf{h}_t$$ puede contener información de toda la historia pasada, no solo de $$n-1$$ tokens.
* No requieren enumerar explícitamente todos los contextos posibles; comparten parámetros entre posiciones y contextos.
* Manejan vocabularios grandes usando embeddings.

En la práctica, RNN simples sufren de gradientes que se desvanecen/explotan; LSTM/GRU mitigan esto y logran capturar mejor dependencias largas [1,8].

---

## 34. Describa las redes LSTM

Las LSTM (Long Short-Term Memory) introducen una célula con memoria interna $$\mathbf{c}_t$$ y puertas para controlar el flujo de información [1,8]:

* **Puerta de olvido** $$\mathbf{f}*t$$: decide qué parte de $$\mathbf{c}*{t-1}$$ se conserva.
* **Puerta de entrada** $$\mathbf{i}_t$$: decide cuánto de la nueva información candidata $$\tilde{\mathbf{c}}_t$$ se integra.
* **Puerta de salida** $$\mathbf{o}_t$$: controla cuánto de $$\mathbf{c}_t$$ se expone como estado oculto $$\mathbf{h}_t$$.

Ecuaciones (simplificadas):

$$
\mathbf{f}*t = \sigma(W_f[\mathbf{h}*{t-1}, \mathbf{x}_t] + b_f)
$$

$$
\mathbf{i}*t = \sigma(W_i[\mathbf{h}*{t-1}, \mathbf{x}_t] + b_i)
$$

$$
\tilde{\mathbf{c}}*t = \tanh(W_c[\mathbf{h}*{t-1}, \mathbf{x}_t] + b_c)
$$

$$
\mathbf{c}_t = \mathbf{f}*t \odot \mathbf{c}*{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
$$

$$
\mathbf{o}*t = \sigma(W_o[\mathbf{h}*{t-1}, \mathbf{x}_t] + b_o)
$$

$$
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
$$

---

## 35. ¿Qué ventajas tienen las LSTM?

* Manejan mejor dependencias largas gracias a la memoria explítica $$\mathbf{c}_t$$ y las puertas que controlan gradientes.
* Reducen problemas de *vanishing gradient* presentes en RNN simples.
* Funcionan bien en tareas secuenciales como LM, traducción, etiquetado secuencial, etc.

---

## 36. ¿Cuál es la diferencia entre una red bidireccional y una unidireccional?

* **Unidireccional**: procesa la secuencia en una sola dirección (izquierda→derecha o derecha→izquierda). El estado en $$t$$ solo ve el pasado (o solo el futuro).
* **Bidireccional**: combina dos redes (p. ej., dos LSTM), una que procesa izquierda→derecha y otra derecha→izquierda; la representación en $$t$$ concatena ambos estados.

Esto permite usar contexto pasado y futuro para representar cada token. Útil en tareas donde toda la secuencia está disponible (tagging, NER, clasificación), pero no directamente para generación autoregresiva [1,8].

---

## 37. ¿Qué es ELMo y cuál es su arquitectura?

ELMo (Embeddings from Language Models) [5] es un modelo contextual basado en LSTM bidireccionales. Arquitectura:

* Un LM bidireccional: una LSTM hacia adelante y otra hacia atrás sobre la secuencia.
* Para cada token, se obtienen representaciones de cada capa y se combinan linealmente (ponderadas) para producir el embedding contextual.

ELMo se preentrena como LM bidireccional y luego se “inyecta” en modelos downstream como características adicionales, manteniendo o no sus parámetros fijos. Fue uno de los primeros éxitos de embeddings contextuales previos a BERT [1,5].

---

## 38. ¿Cómo diseñarías un problema de clasificación con ELMo?

Esquema:

1. Preprocesar texto y obtener embeddings ELMo por token.
2. Agregar los embeddings de todos los tokens (promedio, atención, LSTM adicional) para obtener una representación de la oración/documento.
3. Alimentar esa representación a una o más capas feed-forward y una capa softmax para predecir la clase.
4. Entrenar el clasificador:

   * Opcionalmente, *fine-tunear* ELMo.
   * O usar ELMo congelado como feature extractor.

---

## 39. ¿Qué es pre-training?

El *pre-training* es la fase de entrenamiento inicial de un modelo grande en una tarea auto-supervisada general usando un corpus masivo sin etiquetas. En NLP típicamente se usan objetivos como:

* Next-token prediction (LM autoregresivo).
* Masked Language Modeling (MLM).
* Objectives de denoising (BART, T5).
* Objetivos discriminativos (ELECTRA).

El modelo aprende representaciones generales del lenguaje (sintaxis, semántica, conocimiento de mundo) que luego se reutilizan mediante *transfer learning* [1,10].

---

## 40. ¿Cuál es el paradigma de pre-training y fine-tuning?

Paradigma estándar en NLP moderno [1,10]:

1. **Pre-training**:

   * Entrenar un modelo grande con un objetivo auto-supervisado (LM, MLM, etc.) sobre un gran corpus genérico o de dominio.
   * Se aprende una “base” del modelo (Transformer encoder, decoder, o encoder-decoder).

2. **Fine-tuning**:

   * Añadir cabezas específicas de tarea (clasificación, QA, tagging).
   * Entrenar en datasets etiquetados específicos, ajustando todos o parte de los parámetros.

Este paradigma explicó gran parte del salto de performance entre 2018–2020 (BERT, GPT, RoBERTa, T5).

---

## 41. ¿Qué diferencia hay entre embeddings estáticos y contextuales?

* **Estáticos**: un solo vector por palabra (p. ej., word2vec, GloVe); no depende de la oración. Ventajas: simples, reutilizables. Desventajas: no manejan polisemia ni contexto local.
* **Contextuales**: el embedding del token depende del contexto completo (ELMo, BERT, GPT); la misma palabra en frases distintas tiene vectores distintos [1,11]. Ventajas: desambiguación de sentido, captura relaciones sintácticas y semánticas finas; suelen ser superiores en la mayoría de tareas.

---

## 42. ¿Qué ventaja tiene un embedding contextual frente a uno estático?

Ventajas clave:

* Desambiguación de palabras polisémicas (*banco* de plaza vs *banco* financiero).
* Captura fenómenos sintácticos (acuerdo, dependencias).
* Representa expresiones multi-palabra y composicionalidad de manera más precisa.
* Mejora performance en casi todas las tareas downstream (NER, QA, clasificación, etc.) [1,11].

---

## 43. ¿Cómo se generan embeddings contextuales?

Típicamente con modelos profundos que procesan secuencias completas:

1. Tokenizar la oración.
2. Mapear cada token a un embedding inicial (lookup).
3. Pasar por capas de modelado contextual (LSTM bidireccionales en ELMo; capas Transformer con self-attention en BERT/GPT).
4. En cada capa y para cada posición se obtiene un vector; se usan las salidas de la última capa, o combinación de capas, como embedding contextual.

En encoders bidireccionales (BERT), cada posición ve tanto pasado como futuro; en decoders autoregresivos (GPT), solo pasado [1,9,11].

---

## 44. Explique el mecanismo de atención: cómo funciona, intuición, etc.

La atención aprende a ponderar dinámicamente qué partes de la secuencia de entrada son más relevantes para procesar un token dado [1,9]. En self-attention (Transformers):

* Cada token se proyecta a tres vectores: *query* (Q), *key* (K) y *value* (V).
* La relevancia entre un token i y j se calcula con el producto escalar $$Q_i K_j^\top$$ (escalado y normalizado con softmax).
* Los *weights* de atención definen cómo combinar los *values* $$V_j$$ para construir una representación contextualizada para i.

Intuición: en vez de usar solo vecinos inmediatos o un estado recurrente, el token “pregunta” al resto de la secuencia qué palabras le resultan informativas, y combina sus representaciones proporcionalmente.

---

## 45. ¿Qué es el mecanismo de atención? ¿Cuáles son sus ventajas?

Es un módulo diferenciable que:

* Toma una secuencia de vectores (tokens).
* Aprende pesos de importancia entre todos los pares (i,j).
* Produce nuevas representaciones como combinación ponderada de los valores [1,9].

Ventajas:

* Captura dependencias de largo alcance sin recurrencia.
* Paralelizable (especialmente en Transformers).
* Interpretabilidad parcial: los pesos de atención pueden mostrar qué palabras influyen en cada decisión.
* Base de arquitecturas SOTA (Transformers, BERT, GPT).

---

## 46. ¿Cómo relacionas el mecanismo de atención con la arquitectura Transformers?

El Transformer se construye como una pila de bloques donde el componente central es el *multi-head self-attention* [9]. Cada bloque contiene:

* Capa de self-attention (multi-cabeza).
* Capa feed-forward posicion-wise.
* Conexiones residuales + normalización de capa.

En encoders, la atención es bidireccional sobre toda la secuencia; en decoders, se enmascara para preservar la causalidad. El éxito de Transformers se debe a reemplazar por completo la recurrencia por atención y operaciones totalmente paralelas [1,9].

---

## 47. ¿Qué ventajas tiene Transformers frente a las redes neuronales recurrentes?

* **Paralelización**: RNN procesa secuencialmente; Transformer permite computar representaciones de todos los tokens en paralelo.
* **Depedencias largas**: self-attention conecta directamente cualquier par de posiciones con un solo paso; en RNN la información debe propagarse paso a paso.
* **Mejor escalabilidad**: entrenamiento eficiente en GPU/TPU con grandes lotes.
* **Arquitectura más simple**: bloques homogéneos, menos dificultades de optimización [1,9,10].

---

## 48. ¿Podés hacer un overview de la arquitectura Transformers?

Un Transformer estándar (Vaswani et al. 2017) [9]:

1. **Input**: embeddings de tokens + embeddings posicionales.
2. **Pila de bloques** (N capas): cada bloque tiene:

   * Multi-head self-attention.
   * Suma residual + LayerNorm.
   * Feed-forward position-wise (dos capas lineales con activación).
   * Suma residual + LayerNorm.
3. **Encoder**: pila de bloques sobre la secuencia de entrada.
4. **Decoder** (en encoder-decoder):

   * Self-attention en la secuencia de salida (enmascarado).
   * Cross-attention sobre las salidas del encoder.
   * Feed-forward.
5. **Cabeza de salida**: p. ej. softmax para LM, proyección para clasificación, etc.

Variantes: encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5, BART).

---

## 49. ¿Cómo soluciona el Transformer la falta de sentido posicional (opuesto a las RNN)?

Las RNN tienen orden implícito por la recurrencia; el Transformer procesa todos los tokens en paralelo, por lo que necesita inyectar información de posición explícita. Esto se hace con *positional embeddings* [9]:

* Se suma a los embeddings de tokens un vector de posición (positional encoding), que puede ser sinusoidal (fijo) o aprendido.
* De esta forma, la atención puede distinguir la posición relativa/absoluta de cada token.

---

## 50. ¿Qué es self-attention?

Es atención donde queries, keys y values provienen de la misma secuencia. Se usa para que cada token incorpore información del resto de la secuencia (incluyéndose a sí mismo) [1,9]. Es el mecanismo central de los Transformers.

---

## 51. ¿Qué son los positional embeddings?

Son vectores que codifican la posición (absoluta o relativa) de cada token en la secuencia [9]. Se suman (o concatenan) a los embeddings de tokens antes de las capas de atención, permitiendo al modelo distinguir el orden de las palabras. Pueden ser:

* **Fijos sinusoidales** (como en el Transformer original).
* **Aprendidos** (tablas de embeddings de posición).
* **Relativos** (codifican distancias en lugar de posiciones absolutas).

---

## 52. ¿Qué es el encoder?

En arquitecturas encoder(-decoder), el encoder es el componente que recibe la secuencia de entrada y produce una representación contextualizada de cada token (o un resumen) [1,9,13]. En un Transformer encoder:

* Aplica varias capas de self-attention bidireccional y feed-forward.
* Produce una matriz de tamaño (longitud × dimensión) que codifica el input para uso posterior (p. ej., por el decoder o por una cabeza de clasificación).

BERT es, esencialmente, un encoder Transformer [11].

---

## 53. Describí el input de BERT

Input típico de BERT [11]:

* Secuencia de hasta dos segmentos (ej. par de oraciones A y B).
* Tokens especiales:

  * `[CLS]` al inicio (embedding usado para clasificación).
  * `[SEP]` para separar secuencias (A y B).
* Para cada posición se construye el embedding de entrada como suma de:

  * Token embedding.
  * Segment embedding (A/B).
  * Positional embedding.

Ejemplo: `[CLS]` Oración_A `[SEP]` Oración_B `[SEP]`.

---

## 54. Diseña una tarea de clasificación con BERT

Ejemplo: *sentiment analysis* de oraciones:

1. Input: `[CLS]` + tokens de la oración + `[SEP]`.
2. Pasar por BERT (encoder-only).
3. Tomar el vector de salida asociado a `[CLS]` (representación de la secuencia).
4. Añadir una capa lineal + softmax para predecir la etiqueta (positivo/negativo, etc.).
5. Entrenar BERT + cabeza (fine-tuning) minimizando cross-entropy.

---

## 55. ¿Cómo harías para anonimizar entidades (personas, direcciones, etc.) de fallos judiciales?

Pipeline típico:

1. **Preprocesamiento**: tokenizar el texto de los fallos.
2. **NER**: entrenar/fine-tunear un modelo (p. ej., BERT encoder con cabeza de etiquetado secuencial) para etiquetar tokens con tipos de entidad (PERSON, ORG, LOC, ADDRESS, etc.).
3. **Anonimización**: reemplazar spans etiquetados por placeholders (`[PERSON]`, `[DIR]`, etc.) o pseudónimos coherentes.
4. **Revisión humana** (ideal) para garantizar anonimización total en contextos sensibles.

Este enfoque combina modelos de lenguaje pre-entrenados (transfer learning) con una tarea supervisada específica (NER).

---

## 56. ¿Qué diferencia hay entre un modelo bidireccional y uno unidireccional? ¿Cómo se manifiesta en términos de attention?

* **Unidireccional (causal)**: cada token solo puede atender a posiciones anteriores. En Transformers, esto se implementa con una máscara triangular que impide mirar el futuro; es típico en decoders autoregresivos (GPT).
* **Bidireccional**: cada token puede atender a todas las posiciones (pasadas y futuras). En encoders tipo BERT, no hay máscara causal; self-attention es completa.

Diferencia práctica: bidireccional es mejor para comprensión (clasificación, NER), unidireccional es necesario para generación paso a paso.

---

## 57. ¿Qué son las conexiones residuales y qué rol juegan en la arquitectura Transformers?

Las conexiones residuales (skip connections) suman la entrada de un bloque a su salida antes de la normalización:

$$
\mathbf{y} = \text{LayerNorm}(\mathbf{x} + \text{SubLayer}(\mathbf{x}))
$$

Permiten:

* Facilitar el flujo de gradientes en redes profundas.
* Mitigar problemas de desaparición de gradientes.
* Permitir que capas adicionales aprendan “correcciones” respecto de una identidad, facilitando el entrenamiento de modelos muy profundos [9].

---

## 58. ¿Qué es un encoder?

(Ya respondido conceptualmente en 52, se recalca brevemente.)

Es el componente que procesa la secuencia de entrada y produce representaciones contextuales ricas de cada token, típicamente mediante self-attention bidireccional (Transformers) o RNN bidireccionales (modelos previos). Se usa en tareas de comprensión y como parte del lado izquierdo en arquitecturas encoder-decoder [1,9,13].

---

## 59. Tengo un problema de clasificación, tengo los datos, ¿cómo lo resolverías?

Esquema general usando modelos pre-entrenados:

1. **Definir el objetivo**: tipo de etiquetas, granularidad, métricas (accuracy, F1).
2. **Preprocesamiento**:

   * Limpiar texto si es necesario.
   * Tokenizar según el modelo elegido (BERT, RoBERTa, etc.).
3. **Modelo base**: elegir un encoder pre-entrenado adecuado al idioma/dominio (p. ej., BETO para español).
4. **Arquitectura**: `[CLS]` + texto → encoder → vector `[CLS]` → capa lineal + softmax.
5. **Entrenamiento**:

   * Fine-tuning con cross-entropy.
   * Validación cruzada o dev set para early stopping y ajuste de hiperparámetros.
6. **Evaluación**: F1 macro/micro, matriz de confusión, etc.
7. **Análisis de errores** y mejoras (regularización, aumento de datos, prompt tuning, etc.).

---

## 60. Tengo un problema de NER, tengo los datos, ¿cómo lo resolverías?

Pipeline con encoder pre-entrenado:

1. **Anotación**: datos etiquetados a nivel de token (BIO/BILOU).
2. **Preprocesamiento**: tokenización subpalabras; mapear etiquetas de tokens a sub-tokens (p. ej., etiquetar solo el primero).
3. **Modelo**: encoder (BERT) + cabeza de clasificación token-level (softmax sobre etiquetas). Opcionalmente, añadir CRF para modelar dependencias entre etiquetas.
4. **Entrenamiento**: minimizar cross-entropy (o log-likelihood de CRF).
5. **Evaluación**: Precision/Recall/F1 a nivel de entidad (span-level).
6. **Despliegue**: usar el modelo para etiquetar y, si se desea, anonimizar o estructurar información.

---

## 61. ¿Qué es una arquitectura decoder-only?

En un Transformer decoder-only, solo se utiliza el bloque decoder, sin encoder explícito [10]. Características:

* Self-attention enmascarado (causal) sobre la secuencia generada.
* Posible uso de embeddings adicionales (posicionales, de tipo de token).
* Objetivo: next-token prediction sobre grandes corpus.

Modelos como GPT, LLaMA, Falcon usan esta arquitectura; el mismo modelo sirve como LM generativo y como “asistente” tras alineamiento.

---

## 62. ¿Cuáles son las ventajas y desventajas de la arquitectura encoder-decoder vs decoder-only?

**Encoder-decoder (p. ej. T5, BART)** [9,10]:

* Ventajas:

  * Natural para tareas seq2seq (traducción, resumen); separa procesamiento de entrada y generación de salida.
  * El encoder puede ser bidireccional; el decoder, autoregresivo.
* Desventajas:

  * Más complejo: dos bloques.
  * Menos directo para usar como LM de propósito general (aunque T5 se entrena como text-to-text).

**Decoder-only (p. ej. GPT)**:

* Ventajas:

  * Arquitectura más simple y unificada.
  * Muy natural para LM y generación.
  * Con el framing adecuado, puede resolver también muchas tareas de comprensión vía *prompting* (in-context learning).
* Desventajas:

  * Menos explícitamente diseñado para tareas puramente discriminativas; comprensión se hace vía generación condicionada.
  * Sin encoder separado, algunas tareas estructuradas pueden ser menos eficientes.

---

## 63. ¿Qué se aprende durante el pretraining?

El modelo aprende:

* **Estadística del lenguaje**: gramática, estilo, collocaciones.
* **Semántica**: asociaciones palabra–palabra, roles, relaciones semánticas básicas.
* **Conocimiento de mundo textual**: hechos, entidades, relaciones presentes en el corpus.
* **Habilidades de razonamiento básico** (limitadas) derivadas de patrones en los datos.

En MLM (BERT), aprende a “reconstruir” tokens enmascarados usando contexto bidireccional; en LM generativo (GPT), aprende a predecir el siguiente token [1,10,11].

---

## 64. ¿Qué es un MLM?

MLM (*Masked Language Model*) es un objetivo de pretraining donde se enmascaran aleatoriamente algunos tokens de entrada y el modelo debe predecir los originales [11]. En BERT:

* ~15 % de los tokens se seleccionan.

  * 80 % se reemplaza por `[MASK]`.
  * 10 % por un token aleatorio.
  * 10 % se dejan igual.
* La pérdida se calcula solo sobre los tokens enmascarados.

Permite entrenar encoders bidireccionales porque el modelo ve contexto izquierdo y derecho al mismo tiempo [11].

---

## 65. ¿Cómo se pre-entrena ELECTRA?

ELECTRA [6] usa un objetivo discriminativo llamado *replaced token detection*:

1. Un generador (pequeño MLM) propone reemplazos para tokens enmascarados.
2. Un discriminador (modelo principal) recibe la secuencia con algunos tokens sustituidos y debe clasificar cada posición como “original” o “reemplazado”.

El discriminador se entrena con una tarea de clasificación binaria por token; esto usa más eficientemente el presupuesto de entrenamiento que predecir solo tokens enmascarados. El modelo final es el discriminador, que se usa luego para tareas downstream.

---

## 66. ¿Cómo se pre-entrena T5?

T5 (“Text-To-Text Transfer Transformer”) [7] utiliza un objetivo de *span corruption* (denoising):

* Se seleccionan spans (segmentos contiguos) de tokens y se reemplazan por un token especial.
* El modelo encoder-decoder recibe el texto “corrupto” como entrada y debe generar la concatenación de los spans ocultos como salida.

Todo se formula como problema text-to-text; pretraining y fine-tuning para cualquier tarea se expresan como entrada textual → salida textual (clasificación, QA, traducción, etc.).

---

## 67. Mencione los hiperparámetros más relevantes de BERT

Algunos hiperparámetros clave [11]:

* Número de capas (depth), p. ej., 12 (BERT-base), 24 (BERT-large).
* Dimensión oculta $$d_\text{model}$$ (768, 1024).
* Número de cabezas de atención.
* Dimensión de la FFN interna.
* Tamaño de vocabulario (WordPiece).
* Máxima longitud de secuencia.
* Tasas de dropout.
* Parámetros de entrenamiento: learning rate, batch size, número de pasos, warmup, etc.

---

## 68. ¿Cómo se preentrena un modelo unidireccional?

Con un objetivo de *next-token prediction* (LM autoregresivo) [1,8,10]:

* Dada una secuencia $$w_1,\dots,w_T$$, el modelo maximiza $$\sum_t \log p(w_t \mid w_{<t})$$.
* Se usa atención causal (Transformers) o RNN/LSTM uni-direccional.
* Durante entrenamiento, se suele usar *teacher forcing*: el modelo recibe la secuencia completa desplazada y predice el siguiente token.

Esto es lo que se hace en GPT, LLaMA y otros modelos decoder-only.

---

## 69. Explicá la arquitectura Transformers, cómo se compara con una RNN, pros and cons

(Resumen condensado de respuestas anteriores.)

* **Arquitectura Transformer**: pila de bloques con self-attention multi-cabeza, feed-forward position-wise, conexiones residuales y normalización. Usa embeddings posicionales y puede ser encoder-only, decoder-only o encoder-decoder [9].

**Comparación con RNN**:

* Pros de Transformer:

  * Paralelización plena en tiempo.
  * Manejo eficiente de dependencias largas.
  * Mejor escalabilidad y rendimiento empírico en NLP.
* Contras / retos:

  * Costo cuadrático en atención respecto de la longitud de secuencia (memoria, tiempo).
  * RNN pueden ser más naturales para streaming muy largo, aunque Transformers eficientes han mitigado esto.

---

## 70. ¿Cómo se pueden evaluar los modelos de lenguaje?

(Ya tratado en 26; se resume.)

* Métricas intrínsecas: perplexity, cross-entropy en corpus de prueba.
* Métricas extrínsecas: performance en tareas downstream (GLUE, SuperGLUE, tareas de la materia).
* Evaluación humana: coherencia, veracidad, toxicidad, estilo.
* Métricas específicas de alineamiento: preferencia humana, tasas de alucinaciones, cumplimiento de instrucciones [10,12].

---

## 71. Describí el proceso de MLM

Reformulación breve (ver 64):

1. Tomar una secuencia de tokens.
2. Seleccionar aleatoriamente un subconjunto (p. ej. 15 %) para enmascarar.
3. Reemplazar cada token seleccionado por `[MASK]`, un token aleatorio o dejarlo igual según proporciones definidas.
4. Pasar la secuencia por el encoder bidireccional.
5. Usar una cabeza de clasificación sobre las posiciones seleccionadas para predecir el token original.
6. Minimizar la suma de pérdidas de esas posiciones.

---

## 72. ¿Qué es una arquitectura encoder-decoder?

Es una arquitectura donde un encoder procesa la entrada y produce representaciones, y un decoder genera la salida condicionada en esas representaciones [1,9,13]. Se usa en tareas seq2seq (traducción, resumen, T5). El decoder puede acceder al encoder vía cross-attention (Transformers) o mediante un vector de contexto (RNN clásicas).

---

## 73. ¿Qué es el pre-training? ¿Qué relación tiene con el transfer learning?

Pretraining: fase inicial donde se entrena un modelo en una tarea genérica auto-supervisada sobre datos masivos sin etiquetas (LM, MLM, denoising) [1,10].

Relación con transfer learning: el modelo pre-entrenado actúa como modelo fuente cuyas representaciones se transfieren a tareas objetivo mediante *fine-tuning* o *adapters*. Es la instancia específica de transfer learning que domina en NLP moderno.

---

## 74. ¿En qué capas se centra el pre-training y en cuáles el fine-tuning?

* **Pre-training**: ajusta todos los parámetros del “bloque base” (capas de embeddings, todas las capas de encoder/decoder, capas de atención y FFN).
* **Fine-tuning**: típicamente ajusta:

  * Capas superiores (más cercanas a la salida).
  * Cabezas de tarea.
  * En algunos setups, también capas inferiores pero con learning rate más bajo.

En técnicas de adaptación ligera (adapters, LoRA), el pretraining fija el modelo base y el fine-tuning entrena solo pequeños módulos adicionales.

---

## 75. ¿Qué tipos de pre-entrenamiento conoces?

En NLP:

* LM autoregresivo (GPT, LLaMA).
* MLM (BERT, RoBERTa).
* Denoising autoencoder (BART, T5).
* Replaced token detection (ELECTRA).
* Objectives multitarea (T5: mezcla de tareas text-to-text).
* Pretraining multimodal (CLIP, Flamingo, etc., aunque excede el curso).

---

## 76. ¿Qué información incorpora el pre-entrenamiento?

Ver 63; en resumen:

* Estadística lingüística (sintaxis, semántica, estilos).
* Correlaciones entre entidades, hechos implícitos en los datos.
* Patrones de razonamiento débil derivados del texto.
* Estereotipos y sesgos presentes en los datos.

---

## 77. Explicá conceptualmente el paradigma pre-training y fine-tuning

Conceptualmente:

1. “Aprendizaje generalista”: el modelo aprende una representación de propósito general del lenguaje y conocimiento del mundo a partir de enormes cantidades de texto no etiquetado (pretraining).
2. “Especialización”: se adapta esa representación a tareas específicas mediante entrenamiento adicional con pocos datos etiquetados (fine-tuning).

Ventaja: separa la parte cara de aprender lenguaje (costosa en cómputo, datos) de la parte más barata de adaptar a muchas tareas distintas.

---

## 78. ¿Qué son los adapters?

Los *adapters* son pequeños módulos entrenables insertados dentro de cada capa de un modelo pre-entrenado, normalmente como un cuello de botella de baja dimensión [6]. Durante el fine-tuning:

* Se congelan los pesos originales del modelo base.
* Solo se entrenan los parámetros de los adapters y eventualmente la cabeza de tarea.

Ventajas: eficiencia en parámetros, posibilidad de mantener un solo modelo base y múltiples adapters por tarea/domino.

---

## 79. ¿Qué es LoRA?

LoRA (*Low-Rank Adaptation*) [6] adapta modelos grandes añadiendo matrices de bajo rango a las matrices de pesos existentes (p. ej., en las proyecciones de atención). En vez de actualizar $$W$$ completo, se aprende una descomposición $$\Delta W = A B$$ donde $$A,B$$ tienen rango bajo.

Durante el fine-tuning:

* $$W$$ se mantiene congelado.
* Solo se entrenan $$A,B$$.

Esto reduce drásticamente el número de parámetros entrenables y permite almacenar múltiples adaptaciones ligeras sobre el mismo modelo base.

---

## 80. ¿Qué es un encoder-decoder?

Idéntico a 72; se resume: arquitectura con un encoder que procesa la entrada y un decoder que genera una salida condicionada en las representaciones del encoder, usada para tareas seq2seq (MT, resumen, T5, BART).

---

## 81. ¿Cómo se pre-entrena cada arquitectura encoder-only, encoder-decoder, decoder-only? Menciona un método para cada una.

* **Encoder-only** (BERT, RoBERTa):

  * Pretraining con MLM +/- NSP (BERT).

* **Encoder-decoder** (T5, BART):

  * Pretraining con objetivos de denoising: span corruption (T5), text infilling + otras corrupciones (BART).

* **Decoder-only** (GPT, LLaMA):

  * Pretraining autoregresivo: next-token prediction sobre texto masivo.

---

## 82. ¿Qué son las capacidades emergentes?

Se refiere a comportamientos o habilidades que aparecen solo cuando el modelo alcanza cierta escala (parámetros, datos, cómputo), y que no estaban presentes en versiones más pequeñas. Ejemplos: razonamiento de pocas etapas, habilidades de programación, traducciones de calidad, etc. [10].

---

## 83. ¿Cómo surgen las capacidades emergentes?

Hipótesis:

* La combinación de mayor capacidad (más parámetros), más datos y entrenamiento más prolongado permite que el modelo aprenda patrones más complejos y generalice de formas no lineales con el tamaño.
* Algunos comportamientos parecen “activar” más allá de ciertos umbrales de escala o calidad de datos.

No hay una teoría cerrada; se estudia empíricamente y mediante análisis de escalado [10].

---

## 84. ¿Cuál es la diferencia entre un modelo base y un instruction-tuned?

* **Modelo base**: modelo pre-entrenado solo con objetivos auto-supervisados (LM, MLM, denoising). No está optimizado explícitamente para seguir instrucciones de usuarios; puede requerir *few-shot* explícito y puede comportarse de manera menos alineada con expectativas humanas.
* **Instruction-tuned**: modelo adicionalmente entrenado con *instruction tuning* (supervisado) en pares (instrucción, respuesta) y, en muchos casos, con RLHF u otros métodos de alineamiento. Es mejor siguiendo instrucciones naturales y más útil como asistente [12].

---

## 85. ¿Qué tareas son más apropiadas para un modelo base y cuáles para un instruction-tuned?

* **Modelo base**:

  * Investigación sobre propiedades intrínsecas del LM (perplexity, capacidad de completado puro).
  * Casos donde se controla estrictamente el prompting y el contexto (p. ej., MT autoregresiva clásica).
* **Instruction-tuned**:

  * Asistentes conversacionales.
  * Herramientas interactivas para usuarios finales.
  * Aplicaciones donde se espera que el modelo interprete instrucciones ambiguas o abiertas.

---

## 86. Describa los tres pasos de entrenamiento para generar un asistente (p. ej., ChatGPT)

Esquema típico (OpenAI, LLaMA 2) [10,12]:

1. **Pretraining** (modelo base): LM autoregresivo sobre texto masivo.
2. **Instruction tuning** supervisado (SFT): entrenar el modelo en conjuntos de pares (instrucción, respuesta ideal) producidos por humanos o modelos auxiliares.
3. **RLHF / alineamiento con preferencias**: aprender un *reward model* a partir de comparaciones humanas y usar RL (PPO) u otros métodos (DPO, rejection sampling) para ajustar el modelo y alinearlo con las preferencias humanas (utilidad, seguridad, estilo).

---

## 87. ¿Qué datos necesito para el entrenamiento de instruction tuning?

Necesitas pares (instrucción, respuesta) de alta calidad:

* Instrucciones representativas de los usos esperados (preguntas, tareas, diálogos).
* Respuestas modelo ideal, escritas por humanos expertos o por modelos más potentes revisados por humanos.
* Diversidad de dominios (programación, escritura, razonamiento, etc.).
* Anotaciones cuidadas sobre seguridad y cumplimiento de políticas cuando sea necesario.

---

## 88. ¿Cómo generarías datos sintéticos para un entrenamiento de instruction tuning para un modelo 8B?

Procedimiento típico:

1. Usar un modelo más grande o más capaz (profesor) para generar muchas muestras (instrucción, respuesta) sobre tareas variadas.
2. Filtrar y limpiar automáticamente (chequeos de calidad, longitud, toxicidad).
3. Opcional: muestreo y revisión humana de un subconjunto para calibrar filtros.
4. Balancear el dataset (tipos de tareas, dificultad).

Esto es *self-instruct* o *distillation* desde un modelo más fuerte [12].

---

## 89. Describa el proceso de entrenamiento de instruction tuning

1. Reunir un dataset de instrucciones y respuestas ideales (humanas o sintéticas).
2. Tratar la tarea como supervised learning estándar: entrada = instrucción (+ contexto), salida = respuesta.
3. Fine-tunear el LM pre-entrenado para minimizar la loss de LM condicionada en la instrucción: maximizar la probabilidad de la respuesta completa dado el prompt.
4. Evaluar en benchmarks de instrucciones y ajustar hiperparámetros.

Instruction tuning por sí solo ya produce mejoras significativas en la capacidad de seguir instrucciones [12].

---

## 90. ¿Qué es in-context learning?

Es la capacidad de un LM grande de aprender a realizar una tarea nueva a partir de ejemplos proporcionados en el prompt, sin actualizar explícitamente sus parámetros [12]. El modelo ve una secuencia del tipo:

`Tarea: ... Ejemplo 1: entrada → salida. Ejemplo 2: entrada → salida. Ahora: entrada_nueva → ?`

y produce una salida coherente, como si hubiera “aprendido” la tarea solo con esos ejemplos. Incluye zero-shot (solo instrucción), one-shot, few-shot.

---

## 91. ¿Qué es zero-shot, few-shot?

* **Zero-shot**: el modelo recibe solo una instrucción en lenguaje natural, sin ejemplos, y debe resolver la tarea.
* **Few-shot**: el prompt incluye algunos ejemplos anotados (1–10 típicamente) que ilustran la tarea, y el modelo generaliza a nuevos casos.

Ambos son modalidades de in-context learning.

---

## 92. ¿Qué son las alucinaciones?

“Alucinación” en LLMs es la producción de contenido que es fluido y plausible pero falso, no soportado por los datos de entrenamiento ni por el contexto proporcionado [10]. Por ejemplo, inventar citas bibliográficas, hechos históricos, APIs inexistentes, etc. Es un problema central de seguridad y confiabilidad.

---

## 93. ¿Cómo se mitigan las alucinaciones?

Estrategias:

* Mejorar datos y objetivos de pretraining (calidad, filtrado).
* Instruction tuning que enfatice honestidad, “no sé”, verificación.
* RLHF con recompensas negativas a respuestas falsas o no verificadas.
* Integración con herramientas externas (RAG, buscadores, bases de conocimiento) y entrenamiento para usar las fuentes.
* Controles en el prompt (pedir citas, pedir pasos de razonamiento, “si no estás seguro, dilo”).

---

## 94. ¿Cómo se pasa de un modelo base a un asistente?

Ver 86; síntesis:

1. Partir de un modelo base pre-entrenado.
2. Hacer instruction tuning supervisado con un dataset de instrucciones y respuestas de alta calidad.
3. Aplicar RLHF (u otros métodos de alineamiento) para ajustar el comportamiento según preferencias humanas (educado, útil, seguro).
4. Opcional: adaptar a dominios específicos con datos adicionales, ajuste de estilo, herramientas, RAG, etc.

---

## 95. ¿Qué tipo de prompts funcionan en los modelos base, zero-shot o few-shot? ¿Por qué?

En modelos base (no instruction-tuned) suele ser más efectivo el *few-shot* estructurado que el zero-shot, porque:

* El modelo no ha sido entrenado explícitamente para interpretar instrucciones en lenguaje natural.
* Ejemplos explícitos en el prompt actúan como “plantilla” estadística, ayudando al modelo a replicar el patrón entrada→salida observado.

En cambio, modelos instruction-tuned suelen manejar bien prompts zero-shot.

---

## 96. ¿Qué es el instruction tuning y cómo se diferencia del pre-training?

Instruction tuning es un fine-tuning supervisado sobre (instrucción, respuesta) que adapta un modelo base a seguir instrucciones de forma alineada con preferencias humanas [12]. Pretraining es auto-supervisado y no usa instrucciones explícitas ni feedback humano directo. Instruction tuning:

* Usa datasets mucho más pequeños.
* Cambia la distribución de tareas hacia “instrucciones open-ended”.

---

## 97. ¿Cómo se pueden coleccionar los datos para Instruction Tuning?

Fuentes:

* Anotación humana directa (crowdsourcing, expertos).
* Conversaciones reales entre usuarios y modelos (limpiadas y curadas).
* Generación sintética (self-instruct) por modelos más grandes.
* Extraer de recursos públicos (documentación, Q&A, etc.), transformándolos en formato instrucción→respuesta.

Siempre con procesos de filtrado, de-duplicación y controles de calidad y seguridad.

---

## 98. ¿Se pueden coleccionar sintéticamente? ¿Cómo?

Sí, ver 88:

* Usar un modelo fuerte (profesor) para generar:

  * Instrucciones: “inventar tareas” en distintos dominios.
  * Respuestas: soluciones detalladas.
* Aplicar filtros automáticos (lengua, longitud, toxicidad) + muestreo para revisión humana.
* Iterar el proceso, mejorando prompts y filtros para generar datasets cada vez mejores (self-instruct, data distillation).

---

## 99. ¿Qué es el RLHF? ¿Para qué sirve?

RLHF (*Reinforcement Learning from Human Feedback*) es un proceso donde se usa feedback humano (preferencias entre respuestas) para entrenar un *reward model* y luego optimizar el LM mediante RL para maximizar esa recompensa [12]. Sirve para:

* Alinear el comportamiento del modelo con preferencias humanas (utilidad, seguridad, estilo).
* Corregir desviaciones del simple objetivo de LM (que premia solo plausibilidad).

---

## 100. ¿Cómo se entrena un Reward Model en el contexto de RLHF? ¿Cuál es su función?

Función: asignar una puntuación escalar a una respuesta generada dado un prompt, reflejando cuán preferible es según criterios humanos.

Entrenamiento:

1. Recoger datos de comparaciones humanas: para un prompt, dos o más respuestas; el anotador elige la preferida.
2. Entrenar el reward model para predecir estas preferencias, p. ej. maximizando la probabilidad de que asigne mayor score a la respuesta preferida (loss tipo Bradley–Terry).
3. El reward model se usa luego como función de recompensa en RL (PPO u otros).

---

## 101. Explica RLHF. ¿Cuál es el rol de PPO? ¿Cómo se implementa?

Proceso RLHF típico [12]:

1. Entrenar un modelo base (LM).
2. Entrenar un reward model como se describió.
3. Inicializar una política $$\pi_\theta$$ con el modelo base.
4. Usar RL (PPO, Proximal Policy Optimization) para ajustar $$\theta$$:

   * Para cada prompt, generar una respuesta con la política actual.
   * Obtener recompensa del reward model (y penalizaciones p. ej. por desviarse demasiado de la política base).
   * Actualizar $$\theta$$ para maximizar recompensa bajo restricciones de *KL* con la política base (para no degradar demasiado el LM).

PPO es un algoritmo de RL que permite actualizar la política de forma estable, limitando cada paso de actualización (clipped objective).

---

## 102. ¿Te acordás cómo se lleva LLaMA 2 desde el modelo base hasta el chat? ¿Algún componente novedoso en relación a ChatGPT?

A alto nivel (según documentación de LLaMA 2) [10]:

1. Pretraining decoder-only en corpus masivo multi-idioma.
2. *Supervised fine-tuning* (SFT) para adaptar a formato de chat (instrucciones, roles system/user/assistant).
3. RLHF con comparaciones humanas y técnicas de rechazo (rejection sampling, etc.).

Componentes particulares:

* Uso de *safety fine-tuning* y filtros específicos.
* Distintos tamaños de modelo (7B, 13B, 70B) con variantes chat y base.

La idea general es similar a ChatGPT, con diferencias en detalles de datos y pipelines de seguridad.

---

## 103. ¿Qué es un reward model?

Modelo entrenado para aproximar una función de utilidad humana sobre respuestas generadas dado un prompt. Recibe (prompt, respuesta) y devuelve un escalar que intenta reflejar la preferencia humana esperada. Se usa como critic en RLHF.

---

## 104. ¿Qué rol juega el rejection sampling en LLaMA?

En pipelines tipo LLaMA 2, *rejection sampling* se usa para seleccionar, entre múltiples respuestas generadas por un modelo (o política), aquellas que maximizan alguna métrica (reward model, heurísticas de seguridad) y descartar el resto [10]. Esto puede ocurrir:

* Durante la generación de datos de entrenamiento (SFT, RLHF).
* En inferencia (opcionalmente) para filtrar respuestas de baja calidad o inseguras.

---

## 105. ¿Qué rol juega el token CLS en BERT?

`[CLS]` es un token especial insertado al inicio de la secuencia en BERT [11]. Su embedding de salida (después de todas las capas del encoder) se usa como representación global de la secuencia, especialmente para tareas de clasificación. La cabeza de clasificación toma ese vector y produce logits sobre clases. Aunque BERT no obliga a usar `[CLS]` siempre, el diseño estándar lo trata como “resumen” de la entrada.

---

## Bibliografía básica

[1] Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing* (3rd ed. draft).
[2] Mikolov, T. et al. (2013). *Efficient Estimation of Word Representations in Vector Space*.
[3] Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*.
[4] Pennington, J., Socher, R., & Manning, C. (2014). *GloVe: Global Vectors for Word Representation*.
[5] Peters, M. E. et al. (2018). *Deep contextualized word representations (ELMo)*.
[6] Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP*; Hu, E. J. et al. (2021). *LoRA*.
[7] Raffel, C. et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)*.
[8] Sutskever, I. et al. (2014). *Sequence to Sequence Learning with Neural Networks*; Graves (2013) LSTM; Radford et al. (2018–2019) GPT.
[9] Vaswani, A. et al. (2017). *Attention Is All You Need*.
[10] Brown, T. et al. (2020). *Language Models are Few-Shot Learners*; Touvron et al. (2023). *LLaMA / LLaMA 2*.
[11] Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
[12] Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*; otros trabajos de RLHF/instruction tuning.
