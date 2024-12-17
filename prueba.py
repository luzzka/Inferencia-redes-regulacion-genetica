from predecir import Predecir
import modulos as Mods

carpeta = 'Bioinfo_ DS/'
archivo_entrada = carpeta + 'deg_001.ged'

clase_predicciones = Predecir(archivo_entrada, 'f', "B&B", "Inf_mutua")

#clase_predicciones.leer_archivo_entrada()
#clase_predicciones.generar_predicciones(Mods.mejor_predictor_bb)
#print(clase_predicciones.dic_predicciones)
#clase_predicciones.generar_validaciones()
#clase_predicciones.mostrar_graficos_validaciones()

clase_predicciones.ejecutar_todo()

clase_predicciones = Predecir(archivo_entrada, 'f', "B&B", "Inf_mutua_penalizada")
clase_predicciones.ejecutar_todo()
