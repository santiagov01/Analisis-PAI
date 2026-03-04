from utils import *
from joblib import Parallel, delayed


# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)


'''
======== ENTRENAMIENTO NPK -- ALL MODELS ===============
'''

# ============ Configure Paths ===================

CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod_all_models_NPK/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
CFG.include_prod = False
CFG.individual_train = True

# ==============================================================0
def entrenar_modelo_por_elemento_all(df_imputed, element, model_name, model_config):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_exclude_prod_all_models_NPK/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
    CFG.include_prod = False
    CFG.individual_train = True
    dir_path = f"{CFG.class_path}{element.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # NOTE: It can be changed by 'train_test_nested_model'
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
        CFG=CFG
    )
    return (model_name, element, resultado)
# ===========================================================
# Ejecutar todos los modelos y elementos en paralelo

if CFG.individual_train:
    all_results_list = Parallel(n_jobs=-1, verbose=10)(
        delayed(entrenar_modelo_por_elemento_all)(
            df_imputed=df_imputed,
            element=element,
            model_name=model_name,
            model_config=model_config
        ) for element in CFG.elements_list
        for model_name, model_config in MODELS_CONFIG.items()
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

#==============================================================
# ======= Permutation Importance ==========================
if CFG.individual_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_NPK(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG, all_models=True)
