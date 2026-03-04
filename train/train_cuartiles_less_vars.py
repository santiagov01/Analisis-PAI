from utils import *
from joblib import Parallel, delayed


def save_shap_analysis_results(all_results, class_path, CFG, percentages=[70, 80], 
                                do_permutation=True, df_imputed=None, best_variables=None, all_models=False):
    """
    Función para guardar resultados de análisis SHAP y variables importantes.
    
    Args:
        all_results: Diccionario con resultados de todos los modelos
        class_path: Ruta donde se guardarán los resultados
        CFG: Objeto de configuración
        percentages: Lista de porcentajes para análisis (default: [70, 80])
        do_permutation: Si se debe calcular permutation importance (default: True)
        df_imputed: DataFrame necesario para permutation importance
        best_variables: Lista de las mejores variables para análisis (default: None)
        all_models: Booleano para indicar si se consideran todos los modelos (default: False)
    """
    # Guardar resultados generales
    save_results_general(all_results, class_path)
    
    # Extraer top features para cada porcentaje
    for percent in percentages:
        percent_decimal = percent / 100
        extract_top_x_percent_features(all_results, percent=percent_decimal, 
                                       class_path=class_path, CFG=CFG)
    
    # Calcular SHAP values por categoría y algoritmo
    dict_class_algrtm_shap = category_algorithm_shap_values(all_results)
    save_shap_category_algorithm_csv(dict_class_algrtm_shap, all_results, class_path, CFG=CFG)
    
    # Calcular y guardar importancia de variables para cada porcentaje
    for percent in percentages:
        percent_decimal = percent / 100
        ranking_vars_class = variable_importance_category_algorithm_shap(
            dict_class_algrtm_shap,
            percentage=percent_decimal, 
            results_models_all=all_results, 
            dir_path=class_path,
            CFG=CFG
        )
        save_top_variable_by_category(
            ranking_vars_class,
            percentage=percent_decimal,
            dir_path=class_path
        )
    
    # Análisis de variables más frecuentes para cuartiles
    if CFG.cuartiles_train:
        for percentage in percentages:
            variables_element = {}
            
            # Ruta de variables más importantes en cada algoritmo
            csv_path = f"{class_path}best_{percentage}_percent_features_weight_2_Quartiles.csv"
            
            # Obtener las más importantes ponderadas
            vars = most_frequent_variables_analysis(
                csv_path, 
                element='Quartiles', 
                percentage=percentage,
                dir_path=class_path
            )
            variables_element['Quartiles'] = vars
            
            # Guardar como JSON
            with open(f'{class_path}most_frequent_variables_{percentage}.json', 'w') as f:
                json.dump(variables_element, f, indent=4)
            
            # Guardar en dataframe CSV
            df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
            df_vars.to_csv(f'{class_path}most_frequent_variables_TOTAL_{percentage}.csv', index=False)
    
    # Permutation Importance
    if do_permutation and CFG.cuartiles_train and df_imputed is not None:
        dir_path_permutation = f'{class_path}permutation_importance/'
        permutation_importance_Quartiles(all_results, df_imputed, 
                                        dir_path_permutation, class_path, CFG=CFG, all_models=all_models, best_variables=best_variables)


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)
# ============== READ BEST VARS ===============================
json_best_variables = f"{CFG.Root}/Resultados/classification_exclude_prod/most_frequent_variables_80.json"
# TODO: Checkear que la ruta existe. Se debió ejecutar el modelo de train anterior.
list_best_vars = read_best_variables(json_best_variables)


'''
    TRAIN CUARTILES - NO NESTED
'''
# ============= Configure Paths ===========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True
CFG.include_prod = False

CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"


# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element, best_vars):
    """Entrena un modelo para un elemento específico."""
    
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False

    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_model'
    resultado = train_test_model(
        df_imputed=df_imputed,
        n_clases=None,  # No utilizar para este experimento
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        CFG=CFG,
        best_variables=best_vars
    )
    return (model_name, element, resultado)

if CFG.cuartiles_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name, #Nombre del algoritmo
            model_config=model_config,
            df_imputed=df_imputed,
            element=None,
            best_vars = list_best_vars
        ) for model_name, model_config in zip(list(MODELS_CONFIG.keys()), [MODELS_CONFIG[m] for m in MODELS_CONFIG.keys()])
    )

    # Organizar resultados en estructura de diccionario
    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"Total de combinaciones: {len(all_results_list)}")


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    
    compare_classification_models(all_results, CFG=CFG)

# Guardar resultados de SHAP y variables importantes
save_shap_analysis_results(all_results, CFG.class_path, CFG, 
                          percentages=[70, 80], 
                          do_permutation=True, 
                          df_imputed=df_imputed,
                          best_variables=list_best_vars)


'''
    TRAIN CUARTILES - NESTED
'''
# ============= Configure Paths ===========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_nested/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True
CFG.include_prod = False

CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"


# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento_less_nested(model_name, model_config, df_imputed, element, best_vars):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_nested/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.include_prod = False

    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_model'
    resultado = train_test_class_nested(
        df_imputed=df_imputed,
        n_clases=None,  # No utilizar para este experimento
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        best_variables=best_vars,
        CFG=CFG
    )
    return (model_name, element, resultado)

if CFG.cuartiles_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento_less_nested)(
            model_name=model_name, #Nombre del algoritmo
            model_config=model_config,
            df_imputed=df_imputed,
            element=None,
            best_vars = list_best_vars
        ) for model_name, model_config in zip(list(MODELS_CONFIG.keys()), [MODELS_CONFIG[m] for m in MODELS_CONFIG.keys()])
    )

    # Organizar resultados en estructura de diccionario
    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"Total de combinaciones: {len(all_results_list)}")


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    
    compare_classification_models(all_results, CFG=CFG)

# Guardar resultados de SHAP y variables importantes
save_shap_analysis_results(all_results, CFG.class_path, CFG, 
                          percentages=[70, 80], 
                          do_permutation=True, 
                          df_imputed=df_imputed,
                          best_variables=list_best_vars)


'''
 3.ALL MODELS - TRAINING WITH LESS VARIABLES (BEST VARIABLES) - CUARTILES
'''
# ========== CONFIGURE DIRECTORY/OPTIONS ==========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_all_models/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True



CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

#============= TRAIN LESS VARIABLES/ WITH ALL MODELS=====================================
# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento_less_all_preds(model_name, model_config, df_imputed, element, best_vars):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_all_models/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.individual_train = False
    CFG.cuartiles_train = True
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"

    dir_path = f"{CFG.class_path}{element}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_model'
    resultado = train_test_model_all_predictions(
        df_imputed=df_imputed,
        n_clases=None,  # No utilizar para este experimento
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=True,
        dir_path=dir_path,
        best_variables=best_vars,
        CFG=CFG
    )
    return resultado
# ============================================
if CFG.cuartiles_train:
    # Entrenar todos los modelos
    all_results = {}
    for model_name, model_config in MODELS_CONFIG.items():


        
        resultado = entrenar_modelo_por_elemento_less_all_preds(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=None,
            best_vars=list_best_vars  # acá se podría pasar solamente mejores variables segun el elemento, solo es una idea.
        )
        
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"Total de combinaciones: {len(all_results)}")


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    compare_classification_models(all_results, CFG=CFG)

# Guardar resultados de SHAP y variables importantes
save_shap_analysis_results(all_results, CFG.class_path, CFG, 
                          percentages=[70, 80], 
                          do_permutation=True, 
                          df_imputed=df_imputed,
                          best_variables=list_best_vars,
                            all_models=True
                            )

'''
 4. All models - Cuartiles - PCA - With best variables
'''

CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_pca/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles.pkl"
CFG.include_prod = False
CFG.individual_train = False
CFG.cuartiles_train = True

def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element, best_vars):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_less_vars_pca/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles.pkl"
    CFG.include_prod = False
    CFG.individual_train = False
    CFG.cuartiles_train = True

    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_model'
    resultado = train_test_model_pca(
        df_imputed=df_imputed,
        n_clases=None,  # No utilizar para este experimento
        model_name=model_name,
        model_config=model_config,
        element=element,
        usar_smote=False,
        mostrar_graficos=True,
        calcular_shap=False,
        dir_path=dir_path,
        n_components=2,
        CFG=CFG,
        best_variables=best_vars
    )
    return (model_name, element, resultado)

if CFG.cuartiles_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name, #Nombre del algoritmo
            model_config=model_config,
            df_imputed=df_imputed,
            element=None,
            best_vars=list_best_vars
        ) for model_name, model_config in zip(list(MODELS_CONFIG.keys()), [MODELS_CONFIG[m] for m in MODELS_CONFIG.keys()])
    )

    # Organizar resultados en estructura de diccionario
    all_results = {}
    for model_name, element, resultado in all_results_list:
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"Total de combinaciones: {len(all_results_list)}")


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)

save_results_general(all_results, CFG.class_path)
