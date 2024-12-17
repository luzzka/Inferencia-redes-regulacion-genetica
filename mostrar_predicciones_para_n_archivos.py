from predecir import Predecir
import modulos as Mods
import os  

# carpeta principal donde estan todos los datos
carpeta = 'Bioinfo_ DS/'

funciones_busqueda = ["B&B"]
funciones_puntaje  = ["Inf_mutua", "Inf_mutua_penalizada"]

numeros_archivos = [0, 1, 2, 3, 998, 999]

# Recorrer todos los archivos con extensión .ged en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith('.ged'):
        # Construir la ruta del archivo .ylu correspondiente
        numero_archivo = archivo.split('_')[1].split('.')[0]  # Extraer el número del archivo
        archivo_real = f'red_{numero_archivo}.ylu'  # Nombre del archivo real

        # Verificar si el archivo real existe
        if archivo_real in os.listdir(carpeta) and int(numero_archivo) in numeros_archivos:
            #print(numero_archivo, archivo)
            archivo_entrada = carpeta + archivo
            archivo_salida = carpeta + archivo_real  # Construir la ruta del archivo .ylu

            # Ejecutar Predecir para cada combinación de funciones_busqueda y funciones_puntaje
            for funcion_busqueda in funciones_busqueda:
                for funcion_puntaje in funciones_puntaje:
                    clase_predicciones = Predecir(archivo_entrada, archivo_salida, funcion_busqueda, funcion_puntaje)
                    clase_predicciones.ejecutar_todo()
