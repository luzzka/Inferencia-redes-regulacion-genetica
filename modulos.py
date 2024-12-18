import numpy as np
from itertools import product, combinations
from collections import defaultdict
import random
from scipy.spatial.distance import hamming
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import heapq

# =========================================================================================================
def calcular_informacion_mutua(matriz_datos, combinacion_variables, objetivo):
    num_filas, num_columnas = matriz_datos.shape
    if objetivo >= num_columnas:
        raise ValueError("El índice del objetivo está fuera de rango.")

    # Extraer las columnas de las combinaciones y del objetivo
    combinacion_datos = matriz_datos[:-1, combinacion_variables]  # Excluir la última fila
    objetivo_datos = matriz_datos[1:, objetivo]  # Desplazar una fila hacia arriba

    # Convertir a una sola dimensión de tuplas para las combinaciones
    combinacion_tuplas = np.array([tuple(row) for row in combinacion_datos])

    # Encontrar todas las combinaciones únicas y sus frecuencias
    valores_unicos, indices = np.unique(combinacion_tuplas, return_inverse=True, axis=0)
    tabla_frecuencias = np.zeros((len(valores_unicos), 2), dtype=int)

    # Contar las frecuencias conjuntas
    for i, valor_objetivo in enumerate(objetivo_datos):
        tabla_frecuencias[indices[i], valor_objetivo] += 1

    # Calcular las probabilidades conjuntas y marginales
    total_frecuencias = np.sum(tabla_frecuencias)
    p_conjunta = tabla_frecuencias / total_frecuencias
    p_marginal_combinacion = np.sum(p_conjunta, axis=1)
    p_marginal_objetivo = np.sum(p_conjunta, axis=0)

    # Calcular la información mutua
    informacion_mutua = 0
    for i in range(len(valores_unicos)):
        for y_valor in range(2):
            p_xy_valor = p_conjunta[i, y_valor]
            if p_xy_valor > 0:
                informacion_mutua += p_xy_valor * np.log2(
                    p_xy_valor / (p_marginal_combinacion[i] * p_marginal_objetivo[y_valor])
                )

    return informacion_mutua


# =========================================================================================================

def calcular_informacion_mutua_penalizada(matriz_datos, combinacion_variables, objetivo, penalizacion_factor=1.0):
    num_filas, num_columnas = matriz_datos.shape
    if objetivo >= num_columnas:
        raise ValueError("El índice del objetivo está fuera de rango.")

    # Extraer las columnas de las combinaciones y del objetivo
    combinacion_datos = matriz_datos[:-1, combinacion_variables]  # Excluir la última fila
    objetivo_datos = matriz_datos[1:, objetivo]  # Desplazar una fila hacia arriba

    # Convertir a una sola dimensión de tuplas para las combinaciones
    combinacion_tuplas = np.array([tuple(row) for row in combinacion_datos])

    # Encontrar todas las combinaciones únicas y sus frecuencias
    valores_unicos, indices = np.unique(combinacion_tuplas, return_inverse=True, axis=0)
    tabla_frecuencias = np.zeros((len(valores_unicos), 2), dtype=int)

    # Contar las frecuencias conjuntas
    for i, valor_objetivo in enumerate(objetivo_datos):
        tabla_frecuencias[indices[i], valor_objetivo] += 1

    # Calcular las probabilidades conjuntas y marginales
    total_frecuencias = np.sum(tabla_frecuencias)
    p_conjunta = tabla_frecuencias / total_frecuencias
    p_marginal_combinacion = np.sum(p_conjunta, axis=1)
    p_marginal_objetivo = np.sum(p_conjunta, axis=0)

    # Calcular la información mutua
    informacion_mutua = 0
    for i in range(len(valores_unicos)):
        for y_valor in range(2):
            p_xy_valor = p_conjunta[i, y_valor]
            if p_xy_valor > 0:
                informacion_mutua += p_xy_valor * np.log2(
                    p_xy_valor / (p_marginal_combinacion[i] * p_marginal_objetivo[y_valor])
                )

    # Penalizar la información mutua
    tamaño_combinacion = len(combinacion_variables)
    penalizacion = penalizacion_factor * tamaño_combinacion
    informacion_mutua_penalizada = informacion_mutua - penalizacion

    return informacion_mutua_penalizada

# =========================================================================================================
# Función para calcular una cota superior de información mutua
def calcular_cota_superior(matriz_datos, combinacion_parcial, objetivo, cache_cotas):
    combinacion_key = tuple(sorted(combinacion_parcial))
    if combinacion_key in cache_cotas:
        return cache_cotas[combinacion_key]

    num_filas = matriz_datos.shape[0]
    valores_objetivo = matriz_datos[:, objetivo]
    p_objetivo = np.bincount(valores_objetivo) / num_filas
    entropia_maxima_objetivo = -np.sum(p_objetivo * np.log2(p_objetivo + 1e-9))
    cache_cotas[combinacion_key] = entropia_maxima_objetivo
    return entropia_maxima_objetivo

# Función para calcular la tabla de conteo para una combinación
def generar_tabla_conteo(matriz_datos, combinacion, objetivo):
    num_clases = len(np.unique(matriz_datos[:, objetivo]))
    tabla_conteo = defaultdict(lambda: np.zeros(num_clases, dtype=int))

    for fila in matriz_datos:
        # Asegurar que fila[combinacion] devuelve los valores esperados como un conjunto de índices
        valores_combinacion = tuple(fila[list(combinacion)])  # Corregido para asegurar múltiples índices
        valor_objetivo = fila[objetivo]
        tabla_conteo[valores_combinacion][valor_objetivo] += 1

    return tabla_conteo


# Función para procesar la tabla de conteo y generar las listas concatenadas
def generar_lista_concatenada(tabla_conteo):
    lista_concatenada = []

    for combinacion, conteos in sorted(tabla_conteo.items()):
        max_frecuencia = np.max(conteos)  # Usar numpy para encontrar el máximo
        candidatos = [indice for indice, valor in enumerate(conteos) if valor == max_frecuencia]

        # Resolver empates
        if len(candidatos) > 1:
            prediccion = "X"  # Marcar con 'X' si hay empate
        else:
            prediccion = str(candidatos[0])  # Tomar el índice como predicción

        lista_concatenada.append(prediccion)

    # Concatenar la lista en un solo string
    return "".join(lista_concatenada)




# =========================================================================================================

#Predice las últimas `filas_a_predecir` de la matriz y calcula la distancia de Hamming con los valores reales.
def predecir_matriz(matrixG, json_predictores, filas_a_predecir=10):
    """
    Predice las últimas `filas_a_predecir` de la matriz y calcula la distancia de Hamming con los valores reales.

    Args:
        matrixG (np.ndarray): Matriz con los datos reales.
        json_predictores (dict): Diccionario con los predictores y cadenas para cada gen.
        filas_a_predecir (int): Número de filas finales a predecir.

    Returns:
        dict: Resultados con predicciones, reales y distancias de Hamming.
    """
    # Inicializar variables
    num_filas, num_genes = matrixG.shape
    resultados = {"predicciones": [], "reales": [], "distancias_hamming": [], "f-score": []}

    # Iterar sobre las filas finales
    for t in range(num_filas - filas_a_predecir, num_filas):
        prediccion_t = []

        for gen_objetivo in range(num_genes):
            # Obtener predictores y cadena de predicción del gen
            gen_data = json_predictores.get(f'gen {gen_objetivo}', {})
            predictores_gen = gen_data.get('combinacion', [])
            cadena = gen_data.get('prediccion', '')

            # Valores de los predictores en T-1
            if predictores_gen:
                valores_predictores = matrixG[t - 1, list(predictores_gen)]
                # Convertir a binario y calcular índice
                valor_binario = ''.join(map(str, valores_predictores))
                indice = int(valor_binario, 2)
            else:
                indice = 0  # Si no hay predictores, asumimos índice 0

            # Recuperar el valor predicho de la cadena
            if indice < len(cadena):
                valor_predicho = cadena[indice]
                # Manejar "X" en la cadena con 50% probabilidad
                if valor_predicho == "X":
                    valor_predicho = random.choice([0, 1])
                else:
                    valor_predicho = int(valor_predicho)
            else:
                valor_predicho = random.choice([0, 1])  # Caso fuera de rango

            prediccion_t.append(valor_predicho)

        # Comparar predicción con valores reales
        reales_t = matrixG[t]
        # depurar
        #print("Predicción:", prediccion_t)
        #print("Reales:", reales_t)
        distancia_hamming = hamming(prediccion_t, reales_t) * num_genes
        f1 = f1_score(reales_t, prediccion_t, average='macro')

        # Guardar resultados
        resultados["predicciones"].append(prediccion_t)
        resultados["reales"].append(reales_t.tolist())
        resultados["distancias_hamming"].append(distancia_hamming)
        resultados["f-score"].append(f1)

        # Actualizar matriz con predicciones (siempre usamos reales en T-1)
        matrixG[t] = reales_t  # Datos reales se mantienen para T-1

    return resultados

# Predice las filas futuras de la matriz utilizando un enfoque de Markov,
def predecir_matriz_markov(matrixG, json_predictores, filas_a_predecir=10):
    """
    Predice las filas futuras de la matriz utilizando un enfoque de Markov,
    donde cada fila predicha se utiliza como base para la siguiente predicción.

    Args:
        matrixG (np.ndarray): Matriz con los datos reales.
        json_predictores (dict): Diccionario con los predictores y cadenas para cada gen.
        filas_a_predecir (int): Número de filas finales a predecir.

    Returns:
        dict: Resultados con predicciones, reales, distancias de Hamming y F1-Scores.
    """
    # Inicializar variables
    num_filas, num_genes = matrixG.shape
    resultados = {
        "predicciones": [],
        "reales": [],
        "distancias_hamming": [],
        "f-score": []
    }

    # Crear una copia de la matriz para trabajar con predicciones
    matrix_pred = matrixG.copy()

    # Iterar sobre las filas a predecir
    for t in range(num_filas - filas_a_predecir, num_filas):
        prediccion_t = []

        for gen_objetivo in range(num_genes):
            # Obtener predictores y cadena de predicción del gen
            gen_data = json_predictores.get(f'gen {gen_objetivo}', {})
            predictores_gen = gen_data.get('combinacion', [])
            cadena = gen_data.get('prediccion', '')

            # Valores de los predictores en T-1 (predichos)
            if predictores_gen:
                valores_predictores = matrix_pred[t - 1, list(predictores_gen)]
                # Convertir a binario y calcular índice
                valor_binario = ''.join(map(str, valores_predictores))
                indice = int(valor_binario, 2)
            else:
                indice = 0  # Si no hay predictores, asumimos índice 0

            # Recuperar el valor predicho de la cadena
            if indice < len(cadena):
                valor_predicho = cadena[indice]
                # Manejar "X" en la cadena con 50% probabilidad
                if valor_predicho == "X":
                    valor_predicho = random.choice([0, 1])
                else:
                    valor_predicho = int(valor_predicho)
            else:
                valor_predicho = random.choice([0, 1])  # Caso fuera de rango

            prediccion_t.append(valor_predicho)

        # Comparar predicción con valores reales
        reales_t = matrixG[t]
        # depurar
        #print("Predicción:", prediccion_t)
        #print("Reales:", reales_t)
        distancia_hamming = hamming(prediccion_t, reales_t) * num_genes
        f_score = f1_score(reales_t, prediccion_t, average="macro")

        # Guardar resultados
        resultados["predicciones"].append(prediccion_t)
        resultados["reales"].append(reales_t.tolist())
        resultados["distancias_hamming"].append(distancia_hamming)
        resultados["f-score"].append(f_score)

        # Actualizar la matriz con la predicción para usar en la siguiente iteración
        matrix_pred[t] = prediccion_t

    return resultados


# =========================================================================================================

# Función optimizada Branch and Bound
def mejor_predictor_bb(matriz_datos, objetivo, max_combinaciones=3, funcion_puntaje = calcular_informacion_mutua):
    total_genes = set(range(matriz_datos.shape[1]))
    #total_genes.remove(objetivo)  # Excluimos el objetivo

    mejor_prediccion = {
        "combinacion": None,
        "informacion_mutua": -1
    }

    # Cola de prioridad para explorar combinaciones (inicia con combinaciones de tamaño 1)
    cola = []
    cache_cotas = {}

    for var in total_genes:
        cota = calcular_cota_superior(matriz_datos, [var], objetivo, cache_cotas)
        heapq.heappush(cola, (-cota, (var,)))

    mejor_tabla_conteo = None

    while cola:
        cota_superior_neg, combinacion_actual = heapq.heappop(cola)
        cota_superior = -cota_superior_neg

        # Podar ramas cuya cota superior sea menor que la mejor información mutua encontrada
        if cota_superior <= mejor_prediccion["informacion_mutua"]:
            continue

        # Generar tabla de conteo para la combinación actual
        tabla_conteo = generar_tabla_conteo(matriz_datos, combinacion_actual, objetivo)

        # Calcular información mutua para la combinación actual
        informacion_mutua = funcion_puntaje(matriz_datos, combinacion_actual, objetivo)

        # Actualizar mejor predicción si encontramos una mejor combinación
        if informacion_mutua > mejor_prediccion["informacion_mutua"]:
            mejor_prediccion = {
                "combinacion": combinacion_actual,
                "informacion_mutua": informacion_mutua
            }
            mejor_tabla_conteo = tabla_conteo

        # Generar nuevas combinaciones si no hemos alcanzado el tamaño máximo
        if len(combinacion_actual) < max_combinaciones and len(combinacion_actual) < 6:
            for nuevo_gen in total_genes - set(combinacion_actual):
                nueva_combinacion = tuple(sorted(combinacion_actual + (nuevo_gen,)))
                nueva_cota = calcular_cota_superior(matriz_datos, nueva_combinacion, objetivo, cache_cotas)
                heapq.heappush(cola, (-nueva_cota, nueva_combinacion))

    # Generar la lista concatenada a partir de la mejor tabla de conteo
    lista_concatenada = generar_lista_concatenada(mejor_tabla_conteo)

    return {
    "combinacion": mejor_prediccion["combinacion"],
    "informacion_mutua": mejor_prediccion["informacion_mutua"],
    "tabla_predicciones": lista_concatenada  # Lista generada con las predicciones concatenadas
}



# =========================================================================================================

# Muestra gráficos comparativos para distancias de Hamming y F-Score
def mostrar_resultados(resultados, resultados_markov, nombre_archivo):
    """
    Muestra gráficos comparativos para distancias de Hamming y F-Score 
    entre 'predecir_matriz' y 'predecir_matriz_markov', incluyendo boxplots y violines en un solo gráfico.
    
    Args:
        resultados (dict): Resultados de la función predecir_matriz.
        resultados_markov (dict): Resultados de la función predecir_matriz_markov.
    
    Returns:
        tuple: Cuatro figuras de los gráficos generados.
    """
    # Extraer métricas
    hamming_predecir = resultados['distancias_hamming']
    hamming_markov = resultados_markov['distancias_hamming']
    f_score_predecir = resultados['f-score']
    f_score_markov = resultados_markov['f-score']
    
    # --- Gráfico de líneas: Distancia de Hamming ---
    fig_hamming = plt.figure(figsize=(15, 4))
    plt.plot(hamming_predecir, label="Predecir Matriz", marker='o')
    plt.plot(hamming_markov, label="Predecir Matriz Markov", marker='s')
    plt.title(f"Evolución de la Distancia de Hamming - {nombre_archivo}")
    plt.xlabel("Tiempos Predichos")
    plt.ylabel("Distancia de Hamming")
    plt.legend()
    plt.grid()

    # --- Gráfico de líneas: F-Score ---
    fig_fscore = plt.figure(figsize=(15, 4))
    plt.plot(f_score_predecir, label="Predecir Matriz", marker='o')
    plt.plot(f_score_markov, label="Predecir Matriz Markov", marker='s')
    plt.title(f"Evolución del F-Score - {nombre_archivo}")
    plt.xlabel("Tiempos Predichos")
    plt.ylabel("F-Score")
    plt.legend()
    plt.grid()

    # Preparar datos para boxplots y violines
    data_hamming = [hamming_predecir, hamming_markov]
    data_fscore = [f_score_predecir, f_score_markov]

    # --- Subplots: Boxplot y Violín para Hamming y F-Score ---
    fig_violin = plt.figure(figsize=(15, 6), constrained_layout=True)
    axes = fig_violin.add_subplot(121)
    sns.violinplot(data=data_hamming, ax=axes, linewidth=1.2, alpha=0.6, palette=["#AEDFF7", "#FFB6C1"])
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["Hamming Predecir", "Hamming Markov"])
    axes.set_title(f"Distribución de Distancias de Hamming - {nombre_archivo}")
    axes.set_ylabel("Distancia de Hamming")
    axes.grid(axis='y')

    axes = fig_violin.add_subplot(122)
    sns.violinplot(data=data_fscore, ax=axes, linewidth=1.2, alpha=0.6, palette=["#AEEDE6", "#FFD1DC"])
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["F-Score Predecir", "F-Score Markov"])
    axes.set_title(f"Distribución del F-Score - {nombre_archivo}")
    axes.set_ylabel("F-Score")
    axes.grid(axis='y')

    plt.suptitle("Violinplot")

    # --- Subplots: Boxplot para Hamming y F-Score ---
    fig_boxplot = plt.figure(figsize=(15, 6), constrained_layout=True)

    # Subplot 1: Hamming
    axes = fig_boxplot.add_subplot(121)
    sns.boxplot(data=data_hamming, ax=axes, width=0.3, palette=["#AEDFF7", "#FFB6C1"])
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["Hamming Predecir", "Hamming Markov"])
    axes.set_title(f"Distribución de Distancias de Hamming - {nombre_archivo}")
    axes.set_ylabel("Distancia de Hamming")
    axes.grid(axis='y')

    # Subplot 2: F-Score
    axes = fig_boxplot.add_subplot(122)
    sns.boxplot(data=data_fscore, ax=axes, width=0.3, palette=["#AEEDE6", "#FFD1DC"])
    axes.set_xticks([0, 1])
    axes.set_xticklabels(["F-Score Predecir", "F-Score Markov"])
    axes.set_title(f"Distribución del F-Score - {nombre_archivo}")
    axes.set_ylabel("F-Score")
    axes.grid(axis='y')

    plt.suptitle("Boxplot")

    return fig_hamming, fig_fscore, fig_violin, fig_boxplot
# =========================================================================================================

def mejor_predictor_exhaustivo_sfs(matriz_datos, objetivo, max_combinaciones=4,
                                   funcion_puntaje=calcular_informacion_mutua):
    total_genes = set(range(matriz_datos.shape[1]))
    mejor_prediccion = {
        "combinacion": (),  # Inicialización segura con tupla vacía
        "informacion_mutua": -1
    }
    mejor_tabla_conteo = None

    # Búsqueda exhaustiva hasta combinaciones de tamaño 2
    for k in [1, 2, 3]:
        for combinacion in combinations(total_genes, k):
            tabla_conteo = generar_tabla_conteo(matriz_datos, combinacion, objetivo)
            informacion_mutua = funcion_puntaje(matriz_datos, combinacion, objetivo)

            # Validar que funcion_puntaje no devuelva None
            if informacion_mutua is not None and informacion_mutua > mejor_prediccion["informacion_mutua"]:
                mejor_prediccion = {
                    "combinacion": combinacion,
                    "informacion_mutua": informacion_mutua
                }
                mejor_tabla_conteo = tabla_conteo

    # Validar si se encontraron combinaciones válidas
    if not mejor_prediccion["combinacion"]:
        print("No se encontraron combinaciones válidas en la búsqueda exhaustiva.")
        return {
            "combinacion": (),
            "informacion_mutua": 0,
            "tabla_predicciones": []
        }

    # SFS: Expandir la mejor combinación actual
    combinacion_actual = set(mejor_prediccion["combinacion"])
    while len(combinacion_actual) < max_combinaciones:
        mejor_candidato = None
        mejor_puntaje = -1

        for nuevo_gen in total_genes - combinacion_actual:
            nueva_combinacion = tuple(sorted(combinacion_actual | {nuevo_gen}))
            tabla_conteo = generar_tabla_conteo(matriz_datos, nueva_combinacion, objetivo)
            informacion_mutua = funcion_puntaje(matriz_datos, nueva_combinacion, objetivo)

            # Validar que funcion_puntaje no devuelva None
            if informacion_mutua is not None and informacion_mutua > mejor_puntaje:
                mejor_puntaje = informacion_mutua
                mejor_candidato = nuevo_gen
                mejor_tabla_conteo = tabla_conteo

        # Si encontramos un mejor candidato, lo agregamos
        if mejor_candidato is not None and mejor_puntaje > mejor_prediccion["informacion_mutua"]:
            combinacion_actual.add(mejor_candidato)
            mejor_prediccion = {
                "combinacion": tuple(sorted(combinacion_actual)),
                "informacion_mutua": mejor_puntaje
            }
        else:
            break  # No hay mejora, detener la búsqueda

    # Generar la lista concatenada a partir de la mejor tabla de conteo
    lista_concatenada = generar_lista_concatenada(mejor_tabla_conteo) if mejor_tabla_conteo else []

    return {
        "combinacion": mejor_prediccion["combinacion"],
        "informacion_mutua": mejor_prediccion["informacion_mutua"],
        "tabla_predicciones": lista_concatenada  # Lista generada con las predicciones concatenadas
    }


# =========================================================================================================

# Lee un archivo de predicciones predicciones reales.
def leer_predicciones_archivo(archivo):
    """
    Lee un archivo de predicciones y genera una matriz booleana basada en las relaciones de predicción.

    :param archivo: Ruta al archivo de predicciones.
    :return: Matriz booleana nxn donde n es el número de genes.
    """
    with open(archivo, 'r') as f:
        lineas = f.readlines()

    # Leer la primera línea para obtener el número de genes
    encabezado = lineas[0]
    num_genes = int(encabezado.split('|')[1])  # Ejemplo: "RANDOM|50|1.0"

    # Inicializar matriz booleana con ceros
    matriz_booleana = np.zeros((num_genes, num_genes), dtype=int)

    # Procesar cada línea (a partir de la segunda)
    for i, linea in enumerate(lineas[1:num_genes + 1]):
        predictores, _ = linea.split('|', 1)  # Ignorar el texto después de '|'

        if predictores.strip() != "NN":  # Si no es NN, hay predictores
            indices_predictores = map(int, predictores.split(','))  # Convertir a enteros
            for predictor in indices_predictores:
                matriz_booleana[i, predictor] = 1  # Marcar relación en la matriz

    return num_genes, matriz_booleana


# =========================================================================================================

# Funcion que crea una matriz booleana que indica las relaciones de predicción entre genes.
def crear_matriz_booleana(mejores_combinaciones: dict, num_genes: int):
    # Inicializar matriz booleana con ceros
    matriz_booleana = np.zeros((num_genes, num_genes), dtype=int)

    # Rellenar la matriz en base a las mejores combinaciones
    for gen_key, valores in mejores_combinaciones.items():
        target_gen = int(gen_key.split(' ')[1])  # Extraer el índice del gen
        if "combinacion" in valores and valores["combinacion"]:
            for predictor in valores["combinacion"]:
                matriz_booleana[target_gen, predictor] = 1

    return matriz_booleana

# =========================================================================================================

# Genera la matriz de confusión y calcula métricas relevantes.
def generar_metricas_predicciones(matriz_resultados_obtenidos, matriz_real):
    """
    Genera la matriz de confusión y calcula métricas relevantes.

    Args:
        matriz_real (np.ndarray): Matriz con los valores reales (ground truth).
        matriz_resultados_obtenidos (np.ndarray): Matriz con las predicciones.

    Returns:
        dict: Diccionario con la matriz de confusión y las métricas (f-score, precisión, recall).
    """
    # Aplanar las matrices para facilitar el cálculo
    y_real = matriz_real.flatten()
    y_pred = matriz_resultados_obtenidos.flatten()

    # Calcular los valores de la matriz de confusión
    TP = np.sum((y_real == 1) & (y_pred == 1))  # Verdaderos Positivos
    FP = np.sum((y_real == 0) & (y_pred == 1))  # Falsos Positivos
    FN = np.sum((y_real == 1) & (y_pred == 0))  # Falsos Negativos
    TN = np.sum((y_real == 0) & (y_pred == 0))  # Verdaderos Negativos

    # Matriz de confusión como lista
    matriz_confusion = [[TP, FP], [FN, TN]]

    # Calcular métricas
    precision = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    fscore = f1_score(y_real, y_pred, zero_division=0)

    # Formato final del resultado
    resultado = {
        "matriz_confusion": matriz_confusion,
        "fscore": round(fscore, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3)
    }

    return resultado


# =========================================================================================================

# =========================================================================================================

# =========================================================================================================

# =========================================================================================================

# =========================================================================================================