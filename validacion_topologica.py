'''
Este código sirve para cuando las predicciones ya están hechas
'''
import os 
import json
import re
import numpy as np
import modulos as Mods



def convertir_json_a_diccionario(archivo_json):
    with open(archivo_json, 'r') as file:
        diccionario = json.load(file)
    return diccionario

def obtener_metricas(json_predicciones, archivo_resultados_reales):
    dic_predicciones = convertir_json_a_diccionario(json_predicciones)
    num_genes, matriz_results_reales = Mods.leer_predicciones_archivo(archivo_resultados_reales)
    matriz_predicciones   = Mods.crear_matriz_booleana(dic_predicciones, num_genes)
    
    # diccionario con los resultados de matriz de confusión, f-score...
    resultados_de_comparacion = Mods.generar_metricas_predicciones(matriz_predicciones, matriz_results_reales)
    return resultados_de_comparacion


def guardar_metricas_mejor_modelo(numero, archivo_resultados_reales):

    carpeta = "predicciones_json"
    # Identificar todos los archivos de predicción relacionados con el número
    archivos_predicciones = [
        archivo for archivo in os.listdir(carpeta)
        if archivo.startswith(f"{numero}_") and archivo.endswith(".json")
    ]

    # Diccionario para almacenar los resultados
    resultados = {}
    mejor_fscore = -1
    mejor_modelo = None

    for archivo_prediccion in archivos_predicciones:
        modelo = archivo_prediccion.split("_", 1)[1].replace(".json", "")
        archivo_completo = os.path.join(carpeta, archivo_prediccion)
        
        # Obtener métricas del archivo actual
        metricas = obtener_metricas(archivo_completo, archivo_resultados_reales)
        fscore = metricas["fscore"]

        # Guardar métricas en el diccionario
        resultados[modelo] = metricas

        # Actualizar el mejor modelo
        if fscore > mejor_fscore:
            mejor_fscore = fscore
            mejor_modelo = modelo

    # Agregar el mejor modelo al diccionario
    resultados["mejor_modelo"] = mejor_modelo

    return resultados


def generar_numero_archivo_predcciones():
    numeros_archivos = set()
    patron = re.compile(r"^(\d+)_")  # Busca el número al inicio del archivo seguido de un guion bajo

    # Recorrer los archivos en la carpeta
    for archivo in os.listdir("predicciones_json"):
        if archivo.endswith(".json"):  # Considerar solo archivos JSON
            match = patron.match(archivo)
            if match:
                numero = int(match.group(1))
                numeros_archivos.add(numero)
    
    return numeros_archivos

def convertir_a_serializable(data):
    """
    Convierte un diccionario o estructura de datos que puede contener tipos
    no serializables por JSON (como int64) a tipos nativos de Python.

    Args:
        data (dict): Diccionario o estructura de datos.

    Returns:
        dict: Diccionario con tipos convertidos a formatos serializables.
    """
    if isinstance(data, dict):
        return {key: convertir_a_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convertir_a_serializable(item) for item in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()  # Convierte arrays de NumPy a listas de Python
    else:
        return data


# ==========================================================================================================
def procesar_archivos_y_guardar_resultados(archivo_salida):
    """
    Busca archivos .ylu en la carpeta de resultados, filtra según los números
    generados por generar_numero_archivo_predcciones, calcula métricas y guarda
    los resultados en un único archivo JSON.

    Args:
        carpeta_resultados (str): Carpeta donde se encuentran los archivos .ylu.
        carpeta_predicciones (str): Carpeta donde se encuentran las predicciones JSON.
        archivo_salida (str): Ruta para guardar el archivo JSON con los resultados.
    """
    # Obtener números de archivos de predicciones
    numeros_archivos = generar_numero_archivo_predcciones()

    # Buscar archivos .ylu cuyo número coincida
    archivos_resultados = [
        archivo for archivo in os.listdir("Bioinfo_ DS")
        if archivo.endswith(".ylu") and int(re.match(r"red_(\d+)", archivo).group(1)) in numeros_archivos
    ]

    # Diccionario para almacenar todos los resultados
    resultados_totales = {}

    # Procesar cada archivo .ylu
    for archivo_resultado in archivos_resultados:
        numero = int(re.match(r"red_(\d+)", archivo_resultado).group(1))
        archivo_completo_resultado = os.path.join("Bioinfo_ DS/", archivo_resultado)

        # Obtener métricas para el número actual
        resultados = guardar_metricas_mejor_modelo(numero, archivo_completo_resultado)

        # Guardar los resultados en el diccionario total
        resultados_totales[numero] = convertir_a_serializable(resultados)

    # Guardar todos los resultados en un archivo JSON
    with open(archivo_salida, 'w') as salida_json:
        json.dump(resultados_totales, salida_json, indent=4)

    print(f"Resultados guardados en {archivo_salida}")


# Ejemplo de uso
archivo_salida = "resultados_totales.json"

procesar_archivos_y_guardar_resultados(archivo_salida)
print("TERMINÓ :)")