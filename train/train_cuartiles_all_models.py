from utils import *
from joblib import Parallel, delayed

# =========== LOAD DATA =================

df_imputed = pd.read_csv(CFG.path_df_imputed_corrected)
df_imputed.columns = clean_feature_names(df_imputed.columns)

# ========== CONFIGURE DIRECTORY/OPTIONS ==========================
CFG.class_path = f'{CFG.Root}/Resultados/classification_cuartiles_all_models/'
os.makedirs(CFG.class_path, exist_ok=True)
CFG.individual_train = False
CFG.cuartiles_train = True
CFG.path_pkl_results_classification = f'{CFG.class_path}class_models_cuartiles_all_models.pkl'
# ============================================
if CFG.cuartiles_train:

    element = None


    dir_path = f"{CFG.class_path}"
    # asegurar que exista la ruta de resultados
    os.makedirs(dir_path, exist_ok=True)  # exist_ok=True evita el error si ya existe

    # Entrenar todos los modelos
    all_results = {}
    for model_name, model_config in MODELS_CONFIG.items():
        
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
            best_variables=None,
            CFG=CFG
        )
        
        if model_name not in all_results:
            all_results[model_name] = []
        all_results[model_name].append(resultado)

    print(f"Total de combinaciones: {len(all_results)}")


    with open(CFG.path_pkl_results_classification, 'wb') as pkl_file:
        pickle.dump(all_results, pkl_file)
    compare_classification_models(all_results, CFG=CFG)

save_results_general(all_results, CFG.class_path)

#============= EXTRACT MOST IMPORTANT VARS ==================================
extract_top_x_percent_features(all_results, percent=0.8, class_path=CFG.class_path, CFG=CFG)
extract_top_x_percent_features(all_results, percent=0.7, class_path=CFG.class_path, CFG=CFG)

# ======= Permutation Importance ==========================
if CFG.individual_train:
    dir_path_permutation = f'{CFG.class_path}permutation_importance/'
    permutation_importance_Quartiles(all_results, df_imputed, dir_path_permutation, CFG.class_path, CFG=CFG, all_models=True)
