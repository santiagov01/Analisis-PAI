import sys
sys.path.append('..')
from utils import *
from joblib import Parallel, delayed

# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

'''
    TRAIN CUARTILES - NO NESTED
'''
# ============= Configure Paths ===========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True
CFG.include_prod = False

CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"


# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element):
    """Entrena un modelo para un elemento específico."""
    
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_exclude_prod/'
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
        CFG=CFG
    )
    return (model_name, element, resultado)

if CFG.cuartiles_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento)(
            model_name=model_name, #Nombre del algoritmo
            model_config=model_config,
            df_imputed=df_imputed,
            element=None
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
save_results_general(all_results, CFG.class_path)

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

if CFG.cuartiles_train:
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
    
        # Ruta de variables más importantes en cada algoritmo
        csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_2_Quartiles.csv"
        #csv_path = f"{dir_path}best_{percentage}_percent_features_Quartiles.csv"
        # Obtener las más importantes ponderadas. Para el modelo específico
        vars = most_frequent_variables_analysis(csv_path, 
                                                element='Quartiles', 
                                                percentage=percentage,
                                                dir_path=dir_path)
        variables_element['Quartiles'] = vars

        # Guardar como JSON
        with open(f'{dir_path}most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(f'{dir_path}most_frequent_variables_TOTAL_{percentage}.csv', index=False)
#==============================================================
# ======= Permutation Importance ==========================
if CFG.cuartiles_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_Quartiles(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG)


'''
    TRAIN CUARTILES - NESTED
'''
# ============= Configure Paths ===========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_exclude_prod_nested/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True
CFG.include_prod = False

CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles_all_models.pkl"


# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento_nested(model_name, model_config, df_imputed, element):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_exclude_prod_nested/'
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
        CFG=CFG
    )
    return (model_name, element, resultado)

if CFG.cuartiles_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento_nested)(
            model_name=model_name, #Nombre del algoritmo
            model_config=model_config,
            df_imputed=df_imputed,
            element=None
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
save_results_general(all_results, CFG.class_path)

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

if CFG.cuartiles_train:
    percentages = [70,80]
    for percentage in percentages:
        variables_element = {}
    
        # Ruta de variables más importantes en cada algoritmo
        csv_path = f"{CFG.class_path}best_{percentage}_percent_features_weight_2_Quartiles.csv"
        #csv_path = f"{dir_path}best_{percentage}_percent_features_Quartiles.csv"
        # Obtener las más importantes ponderadas. Para el modelo específico
        vars = most_frequent_variables_analysis(csv_path, 
                                                element='Quartiles', 
                                                percentage=percentage,
                                                dir_path=dir_path)
        variables_element['Quartiles'] = vars

        # Guardar como JSON
        with open(f'{dir_path}most_frequent_variables_{percentage}.json', 'w') as f:
            json.dump(variables_element, f, indent=4)

        # guardar en dataframe csv
        df_vars = pd.DataFrame.from_dict(variables_element, orient='index').transpose()
        df_vars.to_csv(f'{dir_path}most_frequent_variables_TOTAL_{percentage}.csv', index=False)
#==============================================================
# ======= Permutation Importance ==========================
if CFG.cuartiles_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_Quartiles(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG)
