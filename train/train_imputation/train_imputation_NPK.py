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
# 12. guardar resultados de SHAP y Permutation en un archivo de resultados final.



# 1. cargar datos
from statistical_analysis.config import CFG


data_clean = load_data_clean(CFG.data_path_clean)

for idx in range(N_ITERATIONS + 1):

    # 2. dividir en train y test
    train_data, test_data = split_train_test(data_clean, test_size=CFG.test_size,
                                                random_state=BASE_SEED + idx)
    # 3 y 4. entrenar modelo de impuacion para datos de train e imputar datos de test
    train_data_imputed, test_data_imputed = impute_data(train_data, test_data)
    for element in CFG.elements:
        
        for model_name, model_config in CFG.MODELS_CONFIG.items():
            # 5. entrenar modelo de clasificacion con datos de train imputados
            model_result = train_model(train_data_imputed, model_name,
                                    model_config)
            # 6. obtener predicciones de test a partir del modelo entrenado en el paso anterior
            predictions = predict_model(test_data_imputed, model_result['model']) 

            # 7. calcular metricas de clasificacion y guardarlas en un archivo de resultados
            metrics = calculate_metrics(test_data_imputed, predictions)
