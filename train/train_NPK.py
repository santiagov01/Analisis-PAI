from utils import *
from joblib import Parallel, delayed


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


'''
======== 1. ENTRENAMIENTO DE NPK - NO NESTED ===============
'''


# ============ Configure Paths ===================

CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
CFG.include_prod = False
CFG.individual_train = True

# ==============================================================0
def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod/'

    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_nested_model'
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
        CFG=CFG
    )
    return (model_name, element, resultado)
# ===========================================================
# Ejecutar todos los modelos y elementos en paralelo

if CFG.individual_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=element
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


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    compare_classification_models(all_results, CFG=CFG)

save_results_general(all_results, CFG.class_path)

#============= EXTRACT MOST IMPORTANT VARS ==================================
extract_top_x_percent_features(all_results, percent=0.8, class_path=CFG.class_path, CFG=CFG)
extract_top_x_percent_features(all_results, percent=0.7, class_path=CFG.class_path, CFG=CFG)

dict_class_algrtm_shap = category_algorithm_shap_values(all_results)
save_shap_category_algorithm_csv(dict_class_algrtm_shap, all_results, CFG.class_path, CFG=CFG)

ranking_vars_class = variable_importance_category_algorithm_shap(dict_class_algrtm_shap,
                                                                 percentage=0.8, 
                                                                 results_models_all=all_results, 
                                                                 dir_path=CFG.class_path,
                                                                 CFG=CFG)
save_top_variable_by_category(ranking_vars_class,
                            percentage=0.8,
                            dir_path=CFG.class_path)

ranking_vars_class = variable_importance_category_algorithm_shap(dict_class_algrtm_shap,
                                                                 percentage=0.7, 
                                                                 results_models_all=all_results, 
                                                                 dir_path=CFG.class_path,
                                                                 CFG=CFG)
save_top_variable_by_category(ranking_vars_class,
                            percentage=0.7,
                            dir_path=CFG.class_path)


dir_path = CFG.class_path
all_results = load_pickle_results(CFG.path_pkl_results_classification)
if CFG.individual_train:
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
        for element in ['Nitrogen', 'Phosphorus', 'Potassium']:
            # Ruta de variables más importantes en cada algoritmo. Valida para SHAP
            csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_3_{element}.csv"
            
            # Obtener las más importantes ponderadas. Para el modelo específico
            vars = most_frequent_variables_analysis(csv_path, 
                                                    element=element, 
                                                    percentage=percentage,
                                                    dir_path=dir_path)
            variables_element[element] = vars

        # Guardar como JSON
        with open(f"{dir_path}" + f'most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(dir_path + f'most_frequent_variables_TOTAL_{percentage}.csv', index=False)
#==============================================================
# ======= Permutation Importance ==========================
if CFG.individual_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG)


'''
======== 2. ENTRENAMIENTO DE NPK - NESTED ===============
'''

'''
# ============ Configure Paths =======================

CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod_nested/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
CFG.include_prod = False
CFG.individual_train = True

# Función wrapper para entrenar un modelo con un elemento específico
# Línea 143: Modificar la función para recibir class_path como parámetro
def entrenar_modelo_por_elemento_nested(model_name, model_config, df_imputed, element, class_path):
    """Entrena un modelo para un elemento específico."""
    dir_path = f"{class_path}{model_name.replace(' ', '_')}/"  # Usar el parámetro
    os.makedirs(dir_path, exist_ok=True)
    
    resultado = train_test_class_nested(
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

# Línea 167: Pasar class_path explícitamente en la llamada Parallel
if CFG.individual_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento_nested)(
            model_name=model_name,
            model_config=model_config,
            df_imputed=df_imputed,
            element=element,
            class_path=CFG.class_path  # <-- Agregar este parámetro
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


    path_classification_results = CFG.path_pkl_results_classification
    with open(path_classification_results, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    compare_classification_models(all_results, CFG=CFG)

save_results_general(all_results, CFG.class_path)

#============= EXTRACT MOST IMPORTANT VARS ==================================
extract_top_x_percent_features(all_results, percent=0.8, class_path=CFG.class_path, CFG=CFG)
extract_top_x_percent_features(all_results, percent=0.7, class_path=CFG.class_path, CFG=CFG)

dict_class_algrtm_shap = category_algorithm_shap_values(all_results)
save_shap_category_algorithm_csv(dict_class_algrtm_shap, all_results, CFG.class_path, CFG=CFG)

ranking_vars_class = variable_importance_category_algorithm_shap(dict_class_algrtm_shap,
                                                                 percentage=0.8, 
                                                                 results_models_all=all_results, 
                                                                 dir_path=CFG.class_path,
                                                                 CFG=CFG)
save_top_variable_by_category(ranking_vars_class,
                            percentage=0.8,
                            dir_path=CFG.class_path)

ranking_vars_class = variable_importance_category_algorithm_shap(dict_class_algrtm_shap,
                                                                 percentage=0.7, 
                                                                 results_models_all=all_results, 
                                                                 dir_path=CFG.class_path,
                                                                 CFG=CFG)
save_top_variable_by_category(ranking_vars_class,
                            percentage=0.7,
                            dir_path=CFG.class_path)

dir_path = CFG.class_path
all_results = load_pickle_results(CFG.path_pkl_results_classification)
if CFG.individual_train:
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
        for element in ['Nitrogen', 'Phosphorus', 'Potassium']:
            # Ruta de variables más importantes en cada algoritmo. Valida para SHAP
            csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_3_{element}.csv"
            
            # Obtener las más importantes ponderadas. Para el modelo específico
            vars = most_frequent_variables_analysis(csv_path, 
                                                    element=element, 
                                                    percentage=percentage,
                                                    dir_path=dir_path)
            variables_element[element] = vars

        # Guardar como JSON
        with open(f"{dir_path}" + f'most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(dir_path + f'most_frequent_variables_TOTAL_{percentage}.csv', index=False)
#==============================================================
# ======= Permutation Importance ==========================
if CFG.individual_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG)

'''