import numpy as np
import json
import os
import matplotlib.pyplot as plt

import modulos as Mods

class Predecir:
    def __init__(self, genes_entrada, predictores_reales, nombre_tipo_busqueda, nombre_tipo_puntaje):
        self.genes_entrada = genes_entrada
        self.predictores_reales = predictores_reales
        self.matriz_genes = None
        self.dic_predicciones = {}
        self.resultados_validaciones = {}
        self.numero_archivo = genes_entrada.split('/')[-1].split('_')[1].split('.')[0]
        self.graficos_validaciones = None
        self.nombre_tipo_busqueda = nombre_tipo_busqueda
        self.nombre_tipo_puntaje = nombre_tipo_puntaje
        self.funcion_busqueda = None
        self.funcion_puntaje = None
        self.nombre_archivo_final = f'{self.numero_archivo}_{self.nombre_tipo_busqueda}_{self.nombre_tipo_puntaje}'
    
        # --------------------------------------------------------------------
        if self.nombre_tipo_busqueda == "B&B":
            self.funcion_busqueda = Mods.mejor_predictor_bb
        # poner el resto de opciones de busqueda

        if self.funcion_puntaje == "Inf_mutua":
            self.funcion_puntaje = Mods.calcular_informacion_mutua
        else:
            self.funcion_puntaje = Mods.calcular_informacion_mutua_penalizada
        # --------------------------------------------------------------------
        


    def leer_archivo_entrada(self):
        try:
            with open(self.genes_entrada, "r") as file:
                # Leer la primera línea para t, n, y e
                first_line = file.readline().strip()
                t, n, e = map(float, first_line.split())  # Cambiar int a float

                # Leer las siguientes líneas como la matriz binaria
                matrixG = []
                for line in file:
                    matrixG.append(list(map(int, line.strip().split())))  # Para la matriz, asumiendo que los valores son enteros

            self.matriz_genes = np.array(matrixG)
        except Exception as e:
            print(f"Error al leer el archivo: {e}")

        
    # =====================================================================================================
    def generar_predicciones(self):
        self.dic_predicciones = {}

        # Ejecutar la función para obtener los mejores predictores para cada gen
        for gen_target in range(self.matriz_genes.shape[1]):  # Suponiendo que tienes la matriz lista
            print(f"  > Procesando gen {gen_target} ...")
            mejor = self.funcion_busqueda(self.matriz_genes, gen_target, 3, self.funcion_puntaje)

            # Almacenar los resultados en el diccionario
            self.dic_predicciones[f'gen {gen_target}'] = {
                'combinacion': mejor["combinacion"],
                'informacion_mutua_maxima': mejor["informacion_mutua"],
                'prediccion': mejor["tabla_predicciones"]
            }

        output_dir = 'predicciones_json'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f'{self.nombre_archivo_final}.json')

        with open(output_file, 'w') as archivo_json:
            json.dump(self.dic_predicciones, archivo_json, indent=4)

        print("Los datos se guardaron correctamente en", output_file) 


    # =====================================================================================================
    def generar_validaciones(self):
        titulo_grafico = self.numero_archivo + '_' + self.nombre_tipo_busqueda + '_' + self.nombre_tipo_puntaje
        self.resultados_validaciones['dinamica_completa'] = Mods.predecir_matriz(self.matriz_genes, self.dic_predicciones, 
                                                                                 int(self.matriz_genes.shape[0]*20/100))
        self.resultados_validaciones['dinamica_markov'] = Mods.predecir_matriz_markov(self.matriz_genes, self.dic_predicciones, 
                                                                                      int(self.matriz_genes.shape[0]*20/100))

        fig_hamming, fig_fscore, fig_violin, fig_boxplot = Mods.mostrar_resultados(self.resultados_validaciones['dinamica_completa'],
                                                                                   self.resultados_validaciones['dinamica_markov'],
                                                                                   titulo_grafico)
        
        # Crear directorios para cada tipo de gráfico
        carpeta_resultados = 'resultados_predicciones'
        carpetas = {
            'hamming': f'{carpeta_resultados}/hamming',
            'fscore': f'{carpeta_resultados}/fscore', 
            'violin': f'{carpeta_resultados}/violin',
            'boxplot': f'{carpeta_resultados}/boxplot',
            'json': f'{carpeta_resultados}/resultados_json'
        }
        
        # Crear todas las carpetas necesarias
        for carpeta in carpetas.values():
            os.makedirs(carpeta, exist_ok=True)
        
        # Guardar cada figura en su respectiva carpeta
        fig_hamming.savefig(f"{carpetas['hamming']}/{self.nombre_archivo_final}.png")
        fig_fscore.savefig(f"{carpetas['fscore']}/{self.nombre_archivo_final}.png")
        fig_violin.savefig(f"{carpetas['violin']}/{self.nombre_archivo_final}.png")
        fig_boxplot.savefig(f"{carpetas['boxplot']}/{self.nombre_archivo_final}.png")

        # Cerrar las figuras para liberar memoria
        plt.close(fig_hamming)
        plt.close(fig_fscore)
        plt.close(fig_violin)
        plt.close(fig_boxplot)

        # Guardar resultados en JSON
        output_file = os.path.join(carpetas['json'], f'{self.nombre_archivo_final}.json')
        with open(output_file, 'w') as archivo_json:
            json.dump(self.resultados_validaciones, archivo_json, indent=4)

        print("Los datos se guardaron correctamente en", output_file)  # Muestra mensaje de éxito

        # guardar en una tupla
        self.graficos_validaciones = (fig_hamming, fig_fscore, fig_violin, fig_boxplot)
        
    
    # =====================================================================================================
    def ejecutar_todo(self):
        print("=================================================")
        print(f"Ejecutando archivo {self.genes_entrada}")
        print(f"Tipo de búsqueda: {self.nombre_tipo_busqueda}")
        print(f"Función criterio: {self.nombre_tipo_puntaje}")
        print("=================================================")
        
        # Verificar si el archivo JSON ya existe
        output_file = os.path.join('predicciones_json', f'{self.nombre_archivo_final}.json')
        if os.path.exists(output_file):
            print(f"El archivo {self.nombre_archivo_final}.json ya existe. No se ejecutará generar_predicciones().")
        else:
            self.leer_archivo_entrada()
            self.generar_predicciones()
            self.generar_validaciones()
        
        output_file = os.path.join('resultados_predicciones', 'resultados_json', f'{self.nombre_archivo_final}.json')
        with open(output_file, 'r') as archivo_json:
            resultados = json.load(archivo_json)
        
        print("--> Resultados desde JSON:")
        print("Distancias de hamming: ", [f"{float(x):.3f}" for x in resultados['dinamica_completa']['distancias_hamming']])
        print("Resultados F1-score:   ", [f"{float(x):.3f}" for x in resultados['dinamica_completa']['f-score']])
        print("--> Validacion de markov:")
        print("Distancias de hamming: ", [f"{float(x):.3f}" for x in resultados['dinamica_markov']['distancias_hamming']])
        print("Resultados F1-score:   ", [f"{float(x):.3f}" for x in resultados['dinamica_markov']['f-score']])
        print("\n\n\n")
        
