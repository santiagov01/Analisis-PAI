from utils import *
from joblib import Parallel, delayed

# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

'''
======== 1. ENTRENAMIENTO DE NPK ===============
'''

# ============ Configure Paths ===================

CFG.class_path = f'{CFG.Root}/Resultados/classification_NPK_pca/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
CFG.include_prod = False
CFG.individual_train = True
# ====================================================

# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element):
    CFG.class_path = f'{CFG.Root}/Resultados/classification_NPK_pca/'
    os.makedirs(CFG.class_path, exist_ok=True)
    CFG.path_pkl_results_classification = f"{CFG.class_path}class_results_individual_elements.pkl"
    CFG.include_prod = False
    CFG.individual_train = True
    """Entrena un modelo para un elemento específico."""
    dir_path = f"{CFG.class_path}{model_name.replace(' ', '_')}/"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

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
        CFG=CFG
    )
    return (model_name, element, resultado)

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


'''
======== 2. ENTRENAMIENTO DE CUARTILES ===============
'''

CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_pca/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.path_pkl_results_classification = f"{CFG.class_path}class_models_cuartiles.pkl"
CFG.include_prod = False
CFG.individual_train = False
CFG.cuartiles_train = True

# Función wrapper para entrenar un modelo con un elemento específico
def entrenar_modelo_por_elemento(model_name, model_config, df_imputed, element):
    """Entrena un modelo para un elemento específico."""
    CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_pca/'
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