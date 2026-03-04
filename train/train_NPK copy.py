from utils import *
from joblib import Parallel, delayed


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


# ============ FUNCIONES AUXILIARES ===================

def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element, class_path, train_func):
    """Entrena un modelo para un elemento específico.
    
    Args:
        train_func: función de entrenamiento (train_test_model o train_test_class_nested)
    """
    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"
    os.makedirs(dir_path, exist_ok=True)
    
    resultado = train_func(
        df_imputed=df_imputed,
        n_clases=None,
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        CFG=CFG
    )
    return (model_name, element, resultado)


def ejecutar_entrenamiento_paralelo(df_imputed, class_path, train_func):
    """Ejecuta el entrenamiento en paralelo para todos los modelos y elementos."""
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=element,
            class_path=class_path,
            train_func=train_func
        ) for model_name, model_config in zip(list(MODELS_CONFIG.keys()), [MODELS_CONFIG[m] for m in MODELS_CONFIG.keys()])
        for element in CFG.elements_list
    )
    
    # Organizar resultados en estructura de diccionario
    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)
    
    print(f"Total de combinaciones: {len(all_results_list)}")
    return all_results


def guardar_y_comparar_resultados(all_results, class_path, path_pkl_results):
    """Guarda los resultados en pickle y genera comparaciones."""
    with open(path_pkl_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    compare_classification_models(all_results, CFG=CFG)
    save_results_general(all_results, class_path)


def extraer_variables_importantes(all_results, class_path):
    """Extrae las variables más importantes usando SHAP."""
    extract_top_x_percent_features(all_results, percent=0.8, class_path=class_path, CFG=CFG)
    extract_top_x_percent_features(all_results, percent=0.7, class_path=class_path, CFG=CFG)
    
    dict_class_algrtm_shap = category_algorithm_shap_values(all_results)
    save_shap_category_algorithm_csv(dict_class_algrtm_shap, all_results, class_path, CFG=CFG)
    
    for percentage in [0.8, 0.7]:
        ranking_vars_class = variable_importance_category_algorithm_shap(
            dict_class_algrtm_shap,
            percentage=percentage,
            results_models_all=all_results,
            dir_path=class_path,
            CFG=CFG
        )
        save_top_variable_by_category(ranking_vars_class, percentage=percentage, dir_path=class_path)


def analizar_variables_frecuentes(class_path, path_pkl_results):
    """Analiza y guarda las variables más frecuentes."""
    all_results = load_pickle_results(path_pkl_results)
    percentages = [70, 80]
    
    for percentage in percentages:
        variables_element = {}
        for element in ['Nitrogen', 'Phosphorus', 'Potassium']:
            csv_path = f"{class_path}best_{percentage}_percent_features_weight_3_{element}.csv"
            vars = most_frequent_variables_analysis(
                csv_path,
                element=element,
                percentage=percentage,
                dir_path=class_path
            )
            variables_element[element] = vars
        
        # Guardar como JSON
        with open(f"{class_path}most_frequent_variables_{percentage}.json", 'w') as f:
            json.dump(variables_element, f, indent=4)
        
        # Guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(class_path + f'most_frequent_variables_TOTAL_{percentage}.csv', index=False)


def calcular_permutation_importance(all_results, df_imputed, class_path):
    """Calcula la permutation importance."""
    dir_path_permutation = f'{class_path}permutation_importance/'
    permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, class_path, CFG=CFG)


def pipeline_completo(df_imputed, class_path_suffix, train_func):
    """Pipeline completo de entrenamiento y análisis.
    
    Args:
        class_path_suffix: sufijo para la ruta de clasificación (ej: 'classification_exclude_prod/')
        train_func: función de entrenamiento (train_test_model o train_test_class_nested)
    """
    # Configurar paths
    CFG.class_path = f'{CFG.Root}/Resultados/{class_path_suffix}'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
    CFG.include_prod = False
    CFG.individual_train = True
    
    # Entrenar modelos
    if CFG.individual_train:
        all_results = ejecutar_entrenamiento_paralelo(df_imputed, CFG.class_path, train_func)
        guardar_y_comparar_resultados(all_results, CFG.class_path, CFG.path_pkl_results_classification)
        extraer_variables_importantes(all_results, CFG.class_path)
        analizar_variables_frecuentes(CFG.class_path, CFG.path_pkl_results_classification)
        calcular_permutation_importance(all_results, df_imputed, CFG.class_path)


# ===========================================================
# EJECUCIÓN DE PIPELINES
# ===========================================================

'''
======== 1. ENTRENAMIENTO DE NPK - NO NESTED ===============
'''
pipeline_completo(df_imputed, 'classification_exclude_prod/', train_test_model)


'''
======== 2. ENTRENAMIENTO DE NPK - NESTED ===============
'''
pipeline_completo(df_imputed, 'classification_exclude_prod_nested/', train_test_class_nested)
