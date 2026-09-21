# 1. cargar datos
# 2. dividir en train y test (Cambio: Funcion para cargar Base de Datos)
# 3. entrenar modelo de impuacion para datos de train (Cambio: Usar solo datos de train y guardar modelo de imputacion)
# 4. obtener datos de test a partir del modelo entrenado en el paso anterior (Cambio: Utilizar modelo de imputacion guardado para esta iteracion)
# 5. entrenar modelo de clasificacion con datos de train imputados (Cambio: Usar solo datos de train y guardar modelo de clasificacion)
# 6. obtener predicciones de test a partir del modelo entrenado en el paso anterior
# 7. calcular metricas de clasificacion y guardarlas en un archivo de resultados 
# 8. repetir este proceso en N_ITERACIONES y guardar los resultados en un archivo de resultados final
# El proceso también se repite para cada componente N,P,K en este caso, por lo que se tendra un archivo de resultados final para cada componente.
# 9. Después de finalizar las iteraciones, calcular cual fue la mejor iteracion en promedio
# 10. a partir de paso anterior, cargar modelos de clasificion (para cada componente en este caso)
# 11. calcular variables con SHAP y Permutation tomando ese modelo del paso anterior. (para cada componente en este caso)
# para shap se debe hacer 10 iteraciones, tomando 100 muestras distintas del conjunto de test (para modelos basados en kernel, los modelos basados en arboles toman todo el dataset).
# repetir el proceso tanto para los modelos de N, P, K
# 12. guardar resultados de SHAP y Permutation en un archivo de resultados final.



# 1. cargar datos
import os
import json
import pickle
import numpy as np
import pandas as pd
from config import CFG, MODELS_CONFIG, setup_logger
from new_utils import (
    impute_data,
    train_test_NPK,
    split_data_NPK,
    save_results_iterations
)
from xai_utils import (
    run_shap_pipeline,
    run_permutation_pipeline,
    extract_and_save_robust_features
)

'''
data_clean = pd.read_csv(CFG.data_path_clean)
historico_resultados = {elem: {} for elem in CFG.elements}

for idx in range(N_ITERATIONS + 1):

    # 2. dividir en train y test
    train_data, test_data = split_data_NPK(data_clean, test_size=CFG.test_size,
                                                random_state=BASE_SEED + idx)
    # 3 y 4. entrenar modelo de impuacion para datos de train e imputar datos de test
    train_data_imputed, test_data_imputed = impute_data(train_data, test_data)
    for element in CFG.elements:

        for model_name, model_config in CFG.MODELS_CONFIG.items():
            # 5. entrenar modelo de clasificacion con datos de train imputados
            # 6. obtener predicciones de test a partir del modelo entrenado en el paso anterior
            # 7. calcular metricas de clasificacion y guardarlas en un archivo de resultados
            model_result = train_test_NPK(train_data_imputed,
                                           test_data_imputed,
                                           element,
                                           model_name,
                                           model_config)
            historico_resultados[element][idx][model_name] = model_result    
            
metrica_objetivo = 'f1_test_macro' 

mejores_iteraciones_info = {}
best_results = {}

for element in CFG.elements:
    mejor_iteracion = -1
    mejor_promedio = -1.0
    
    # Evaluar cada iteración registrada para este elemento
    for idx in range(N_ITERATIONS + 1):
        results_iter = historico_resultados[element][idx]
        
        # Extraer la métrica de interés de todos los modelos entrenados en esta iteración
        metricas_modelos = [
            resultado[metrica_objetivo] 
            for modelo, resultado in results_iter.items()
        ]
        
        # Calcular el promedio de los modelos para esta iteración
        promedio_iteracion = np.mean(metricas_modelos)
        
        # Comparar y actualizar si es el mejor hasta ahora
        if promedio_iteracion > mejor_promedio:
            mejor_promedio = promedio_iteracion
            mejor_iteracion = idx
            
    # Guardar los metadatos de la iteración ganadora
    mejores_iteraciones_info[element] = {
        'iteracion': mejor_iteracion,
        f'{metrica_objetivo}_promedio': mejor_promedio
    }
    
    # Aislar y guardar LOS RESULTADOS COMPLETOS de la iteración ganadora para este elemento
    best_results[element] = historico_resultados[element][mejor_iteracion]

save_results_iterations(historico_resultados,
                        best_results,
                        mejores_iteraciones_info)

del historico_resultados


'''











def main():
    # 0. Inicializar Directorios y Logger
    # NOTE: Se puede configurar el directorio destino de guardar los datos.
    os.makedirs(CFG.xai_output_dir, exist_ok=True)
    log_file = os.path.join(CFG.results_dir, "train_imputation_NPK.log")
    logger = setup_logger(name="Pipeline_NPK", log_file=log_file)
    
    logger.info("Iniciando pipeline de clasificación e interpretabilidad NPK")

    checkpoint_file = os.path.join(CFG.results_dir, "checkpoint_train_imputation_NPK.json")
    progress_file = os.path.join(CFG.results_dir, "historico_resultados_progress.pkl")

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as file:
            checkpoint = json.load(file)
    else:
        checkpoint = {'completadas': []}

    completed_iterations = set(checkpoint.get('completadas', []))

    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as file:
            historico_resultados = pickle.load(file)
    else:
        historico_resultados = {elem: {} for elem in CFG.elements_list}
    
    # 1. Carga de Base de Datos
    logger.info(f"Cargando dataset desde: {CFG.data_path_clean}")
    data_clean = pd.read_csv(CFG.data_path_clean)
    
    # 2. Bucle de Partición, Imputación y Entrenamiento
    logger.info(f"Iniciando evaluación en {CFG.n_iterations} iteraciones")
    
    for idx in range(CFG.n_iterations):
        current_seed = CFG.base_seed + idx
        iteration_key = f"iteration_{idx}_seed_{current_seed}"

        iteration_saved = all(
            idx in historico_resultados[element]
            for element in CFG.elements_list
        )
        if iteration_key in completed_iterations and iteration_saved:
            logger.info(f"Saltando iteración {idx + 1}/{CFG.n_iterations}: ya está registrada")
            continue

        logger.info(f"Ejecutando iteración {idx + 1}/{CFG.n_iterations} (Semilla: {current_seed})")
        
        train_data, test_data = split_data_NPK(
            data_clean, 
            test_size=CFG.test_size, 
            random_state=current_seed
        )
        
        train_data_imputed, test_data_imputed = impute_data(train_data, test_data, seed=current_seed)
        
        for element in CFG.elements_list:
            if idx not in historico_resultados[element]:
                historico_resultados[element][idx] = {}
                
            for model_name, model_config in MODELS_CONFIG.items():
                logger.info(f"Modelo '{model_name}' - elemento '{element}' - Iteración {idx + 1}")
                model_result = train_test_NPK(
                                train_data_imputed,
                                test_data_imputed,
                                element,
                                model_name,
                                model_config,
                                seed=current_seed)
                
                historico_resultados[element][idx][model_name] = model_result

        completed_iterations.add(iteration_key)
        with open(progress_file, 'wb') as file:
            pickle.dump(historico_resultados, file)
        with open(checkpoint_file, 'w', encoding='utf-8') as file:
            json.dump({'completadas': sorted(completed_iterations)}, file, indent=4)
        logger.info(f"Checkpoint guardado: {iteration_key}")

                
    # 3. Selección de la Mejor Iteración por Elemento
    logger.info(f"Calculando la mejor iteración por elemento según métrica '{CFG.target_metric}'")
    mejores_iteraciones_info = {}
    best_results = {}
    
    for element in CFG.elements_list:
        mejor_iteracion = -1
        mejor_promedio = -1.0
        
        for idx in range(CFG.n_iterations):
            results_iter = historico_resultados[element][idx]
            # Extraer la métrica de interés de todos los modelos entrenados en esta iteración
            metricas = [res[CFG.target_metric] for _, res in results_iter.items()]
            promedio = float(np.mean(metricas))

            # Comparar y actualizar si es el mejor promedio hasta ahoras
            if promedio > mejor_promedio:
                mejor_promedio = promedio
                mejor_iteracion = idx
        # Guardar los metadatos de la iteración ganadora
        mejores_iteraciones_info[element] = {
            'mejor_iteracion': mejor_iteracion,
            f'{CFG.target_metric}_promedio': mejor_promedio
        }
        # Aislar y guardar los resultados completos de la iteración ganadora para este elemento
        best_results[element] = historico_resultados[element][mejor_iteracion]
        logger.info(f"Elemento '{element}': Ganadora Iteración {mejor_iteracion + 1} (Score Promedio: {mejor_promedio:.4f})")
    # Guardar resultados del entrenamiento
    save_results_iterations(historico_resultados, best_results, mejores_iteraciones_info,
                            base_path=CFG.results_dir)
    del historico_resultados

    # 4. Flujo de Interpretabilidad SHAP
    logger.info("Ejecutando pipeline de interpretabilidad SHAP")
    shap_results = run_shap_pipeline(
        best_results=best_results,
        elements=CFG.elements_list,
        models_config=MODELS_CONFIG,
        output_dir=CFG.xai_output_dir,
        n_iterations=CFG.shap_iterations,
        base_seed=CFG.base_seed,
        logger=logger
    )
    # 5. Flujo de Interpretabilidad Permutation Importance
    logger.info("Ejecutando pipeline de interpretabilidad Permutation Importance")
    perm_results = run_permutation_pipeline(
        best_results=best_results,
        elements=CFG.elements_list,
        models_config=MODELS_CONFIG,
        output_dir=CFG.xai_output_dir,
        base_seed=CFG.base_seed,
        logger=logger
    )
    # 6. Consolidación Final: Intersección SHAP + Permutation
    logger.info("Consolidando variables robustas finales (Intersección SHAP y Permutation)")
    extract_and_save_robust_features(
        shap_results=shap_results,
        perm_results=perm_results,
        elements=CFG.elements_list,
        output_dir=CFG.xai_output_dir,
        logger=logger
    )
    logger.info("Pipeline completado exitosamente. Todos los archivos han sido generados.")
if __name__ == '__main__':
    main()